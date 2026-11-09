// End-to-end check against the NumPy reference: every layer's output, the
// last-position logits and the greedy continuation of each golden prompt.
// Golden files come from `tools/reference: uv run qwen35-ref dump`.

#include "io/safetensors.h"
#include "model/config.h"
#include "model/weights.h"
#include "runtime/executor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Relative error bound per layer. Both sides are f32 with a different summation
// order; the naive kernels land at ~4e-6, so this leaves 5x headroom.
constexpr double kTolerance = 2e-5;

const fs::path kCheckpoint = MINI_LLM_DEFAULT_WEIGHTS;
const fs::path kGolden = MINI_LLM_GOLDEN_DIR;

struct Golden {
    std::string name;
    std::map<std::string, std::vector<float>> f32;
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> greedy_ids;
};

Golden read_golden(const fs::path& file) {
    const io::SafetensorsFile st(file.string());
    Golden g{file.stem().string(), {}, {}, {}};
    for (const io::TensorEntry& e : st.entries()) {
        if (e.dtype == core::DType::i32) {
            std::vector<std::int32_t> v(e.bytes / sizeof(std::int32_t));
            std::memcpy(v.data(), e.data, e.bytes);
            (e.name == "input_ids" ? g.input_ids : g.greedy_ids) = std::move(v);
        } else {
            std::vector<float> v(e.bytes / sizeof(float));
            std::memcpy(v.data(), e.data, e.bytes);
            g.f32[e.name] = std::move(v);
        }
    }
    return g;
}

double relative_error(std::span<const float> got, const std::vector<float>& want) {
    double diff = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        diff = std::max(diff, std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
        scale = std::max(scale, std::abs(static_cast<double>(want[i])));
    }
    return diff / (scale + 1e-12);
}

class GoldenTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!fs::exists(kCheckpoint / "config.json") || !fs::exists(kGolden)) {
            return;
        }
        config_ = std::make_unique<model::Config>(model::Config::load(kCheckpoint));
        checkpoint_ = std::make_unique<io::Checkpoint>(kCheckpoint.string());
        weights_ = std::make_unique<model::Weights>(model::Weights::load(*config_, *checkpoint_));
        for (const auto& entry : fs::directory_iterator(kGolden)) {
            if (entry.path().extension() == ".safetensors") {
                goldens_.push_back(read_golden(entry.path()));
            }
        }
        std::ranges::sort(goldens_, {}, &Golden::name);
    }

    static void TearDownTestSuite() {
        weights_.reset();
        checkpoint_.reset();
        config_.reset();
        goldens_.clear();
    }

    void SetUp() override {
        if (weights_ == nullptr || goldens_.empty()) {
            GTEST_SKIP() << "needs the checkpoint and golden files (scripts/download_weights.sh, qwen35-ref dump)";
        }
    }

    static runtime::Executor executor(std::size_t max_chunk) {
        return runtime::Executor(*config_, *weights_, {.max_seq = 512, .max_chunk = max_chunk});
    }

    static inline std::unique_ptr<model::Config> config_;
    static inline std::unique_ptr<io::Checkpoint> checkpoint_;
    static inline std::unique_ptr<model::Weights> weights_;
    static inline std::vector<Golden> goldens_;
};

TEST_F(GoldenTest, EveryLayerMatchesTheReference) {
    for (const Golden& g : goldens_) {
        SCOPED_TRACE(g.name);
        runtime::Executor ex = executor(256);
        std::map<std::string, double> errors;
        const runtime::Trace trace = [&](std::string_view name, std::span<const float> data, std::size_t, std::size_t) {
            const auto it = g.f32.find(std::string(name));
            ASSERT_NE(it, g.f32.end()) << name;
            ASSERT_EQ(data.size(), it->second.size()) << name;
            errors[std::string(name)] = relative_error(data, it->second);
        };
        ex.forward(g.input_ids, nullptr, &trace);
        EXPECT_EQ(errors.size(), g.f32.size());
        for (const auto& [name, err] : errors) {
            EXPECT_LT(err, kTolerance) << name;
        }
    }
}

TEST_F(GoldenTest, GreedyDecodeMatchesTheReference) {
    for (const Golden& g : goldens_) {
        SCOPED_TRACE(g.name);
        runtime::Executor ex = executor(256);
        std::vector<std::int32_t> out{ex.forward(g.input_ids)};
        while (out.size() < g.greedy_ids.size()) {
            out.push_back(ex.forward(std::span(&out.back(), 1)));
        }
        EXPECT_EQ(out, g.greedy_ids);
    }
}

TEST_F(GoldenTest, ChunkedPrefillMatchesTheReference) {
    for (const Golden& g : goldens_) {
        SCOPED_TRACE(g.name);
        runtime::Executor ex = executor(4);
        std::vector<float> logits;
        const std::int32_t next = ex.forward(g.input_ids, &logits);
        EXPECT_LT(relative_error(logits, g.f32.at("logits")), kTolerance);
        EXPECT_EQ(next, g.greedy_ids.front());
    }
}

TEST_F(GoldenTest, ResetStartsANewSequence) {
    const Golden& g = goldens_.front();
    runtime::Executor ex = executor(256);
    ex.forward(g.input_ids);
    ex.reset();
    std::vector<float> logits;
    ex.forward(g.input_ids, &logits);
    EXPECT_EQ(ex.position(), g.input_ids.size());
    EXPECT_LT(relative_error(logits, g.f32.at("logits")), kTolerance);
}

} // namespace
