// Greedy generation from token ids, with timing.
//
//   generate [--checkpoint DIR] [--max-new N] [ID ...]
//
// Without ids it uses "The capital of France is" (760 6511 314 9338 369).
// Tokenization arrives in Phase 7; until then use `qwen35-ref` to encode text.

#include "io/safetensors.h"
#include "model/config.h"
#include "model/weights.h"
#include "runtime/executor.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path checkpoint = "weights/qwen3.5-0.8b";
    std::size_t max_new = 32;
    std::vector<std::int32_t> ids;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--checkpoint" && i + 1 < argc) {
                checkpoint = argv[++i];
            } else if (arg == "--max-new" && i + 1 < argc) {
                max_new = std::stoul(argv[++i]);
            } else {
                ids.push_back(static_cast<std::int32_t>(std::stol(std::string(arg))));
            }
        }
        if (ids.empty()) {
            ids = {760, 6511, 314, 9338, 369};
        }

        const model::Config config = model::Config::load(checkpoint);
        const io::Checkpoint ckpt(checkpoint.string());
        const model::Weights weights = model::Weights::load(config, ckpt);
        runtime::Executor ex(config, weights, {.max_seq = ids.size() + max_new, .max_chunk = 256});

        auto t0 = Clock::now();
        std::vector<std::int32_t> out{ex.forward(ids)};
        const double prefill_ms = ms_since(t0);

        t0 = Clock::now();
        while (out.size() < max_new && out.back() != config.eos_token_id) {
            out.push_back(ex.forward(std::span(&out.back(), 1)));
        }
        const double decode_ms = ms_since(t0);
        const std::size_t steps = out.size() - 1;

        std::print("ids");
        for (std::int32_t id : out) {
            std::print(" {}", id);
        }
        std::println("");
        std::println("prefill  {} tokens in {:.2f} ms", ids.size(), prefill_ms);
        if (steps > 0) {
            std::println("decode   {} tokens, {:.3f} ms/token ({:.0f} tok/s)", steps,
                         decode_ms / static_cast<double>(steps), 1e3 * static_cast<double>(steps) / decode_ms);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "generate: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
