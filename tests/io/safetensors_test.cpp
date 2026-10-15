#include "io/safetensors.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// A fresh directory under the system temp dir, removed on destruction.
class TempDir {
public:
    TempDir()
        : path_(fs::temp_directory_path() / ("mini_llm_" + std::to_string(counter_++) + "_" +
                                             ::testing::UnitTest::GetInstance()->current_test_info()->name())) {
        fs::create_directories(path_);
    }
    ~TempDir() { fs::remove_all(path_); }
    const fs::path& path() const { return path_; }

private:
    static inline int counter_ = 0;
    fs::path path_;
};

void write_safetensors(const fs::path& file, const std::string& header, const std::vector<std::uint8_t>& data) {
    std::ofstream out(file, std::ios::binary);
    const std::uint64_t len = header.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

TEST(SafetensorsFile, MapsTensors) {
    TempDir dir;
    const fs::path file = dir.path() / "model.safetensors";
    std::vector<std::uint8_t> data(8 + 4);
    const float values[2] = {1.5f, -2.0f};
    std::memcpy(data.data(), values, sizeof(values));
    write_safetensors(file,
                      R"({"__metadata__": {"format": "pt"},
                          "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                          "h": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [8, 12]}})",
                      data);

    const io::SafetensorsFile st(file.string());
    ASSERT_EQ(st.entries().size(), 2u);
    const io::TensorEntry& w = st.entries()[0];
    EXPECT_EQ(w.name, "w");
    EXPECT_EQ(w.dtype, core::DType::f32);
    EXPECT_EQ(w.shape, (std::vector<std::size_t>{2}));
    float read[2];
    std::memcpy(read, w.data, sizeof(read));
    EXPECT_EQ(read[0], 1.5f);
    EXPECT_EQ(read[1], -2.0f);
    EXPECT_EQ(st.entries()[1].dtype, core::DType::bf16);
    EXPECT_EQ(st.entries()[1].bytes, 4u);
}

TEST(SafetensorsFile, RejectsInconsistentEntries) {
    TempDir dir;
    const fs::path file = dir.path() / "bad.safetensors";
    const std::vector<std::uint8_t> data(8);
    for (const char* header : {
             R"({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 16]}})", // past end of data
             R"({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}})",  // size != shape
             R"({"w": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}})",  // unsupported dtype
             R"({"w": {"dtype": "F32", "shape": [2]}})",                          // missing offsets
         }) {
        write_safetensors(file, header, data);
        EXPECT_THROW(io::SafetensorsFile{file.string()}, std::runtime_error) << header;
    }
}

TEST(SafetensorsFile, RejectsHeaderLongerThanFile) {
    TempDir dir;
    const fs::path file = dir.path() / "short.safetensors";
    {
        std::ofstream out(file, std::ios::binary);
        const std::uint64_t len = 1000;
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    }
    EXPECT_THROW(io::SafetensorsFile{file.string()}, std::runtime_error);
}

TEST(Checkpoint, ResolvesShardsThroughIndex) {
    TempDir dir;
    const std::vector<std::uint8_t> data(4);
    write_safetensors(dir.path() / "a.safetensors", R"({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})",
                      data);
    write_safetensors(dir.path() / "b.safetensors", R"({"y": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})",
                      data);
    std::ofstream(dir.path() / "model.safetensors.index.json")
        << R"({"metadata": {}, "weight_map": {"x": "a.safetensors", "y": "b.safetensors"}})";

    const io::Checkpoint ckpt(dir.path().string());
    EXPECT_NE(ckpt.find("x"), nullptr);
    EXPECT_NE(ckpt.find("y"), nullptr);
    EXPECT_EQ(ckpt.find("z"), nullptr);
    EXPECT_THROW(ckpt.at("z"), std::runtime_error);
    ASSERT_EQ(ckpt.entries().size(), 2u);
    EXPECT_EQ(ckpt.entries()[0]->name, "x");
}

TEST(Checkpoint, RejectsTensorInTwoShards) {
    TempDir dir;
    const std::vector<std::uint8_t> data(4);
    const char* header = R"({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})";
    write_safetensors(dir.path() / "a.safetensors", header, data);
    write_safetensors(dir.path() / "b.safetensors", header, data);
    std::ofstream(dir.path() / "model.safetensors.index.json")
        << R"({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}})";
    EXPECT_THROW(io::Checkpoint{dir.path().string()}, std::runtime_error);
}

} // namespace
