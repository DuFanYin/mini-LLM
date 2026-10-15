#include "io/json.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using io::Json;

TEST(Json, ParsesNestedDocument) {
    const Json j = Json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": true, "d": null}, "e": "x"})");
    const auto& a = j["a"].as_array();
    ASSERT_EQ(a.size(), 3u);
    EXPECT_EQ(a[0].as_int(), 1);
    EXPECT_DOUBLE_EQ(a[1].as_number(), 2.5);
    EXPECT_DOUBLE_EQ(a[2].as_number(), -300.0);
    EXPECT_TRUE(j["b"]["c"].as_bool());
    EXPECT_TRUE(j["b"]["d"].is_null());
    EXPECT_EQ(j["e"].as_string(), "x");
}

TEST(Json, KeepsObjectKeyOrder) {
    const Json j = Json::parse(R"({"z": 1, "a": 2, "m": 3})");
    const auto& o = j.as_object();
    ASSERT_EQ(o.size(), 3u);
    EXPECT_EQ(o[0].first, "z");
    EXPECT_EQ(o[1].first, "a");
    EXPECT_EQ(o[2].first, "m");
}

TEST(Json, DecodesEscapesToUtf8) {
    const Json j = Json::parse(R"(["\n\t\"\\\/", "é", "中", "😀"])");
    const auto& a = j.as_array();
    EXPECT_EQ(a[0].as_string(), "\n\t\"\\/");
    EXPECT_EQ(a[1].as_string(), "\xC3\xA9");
    EXPECT_EQ(a[2].as_string(), "\xE4\xB8\xAD");
    EXPECT_EQ(a[3].as_string(), "\xF0\x9F\x98\x80");
}

TEST(Json, FindReturnsNullForMissingKey) {
    const Json j = Json::parse(R"({"a": 1})");
    EXPECT_NE(j.find("a"), nullptr);
    EXPECT_EQ(j.find("b"), nullptr);
    EXPECT_THROW(j["b"], std::runtime_error);
}

TEST(Json, RejectsMalformedInput) {
    for (const char* bad : {"", "{", "[1,]", R"({"a" 1})", "tru", "1 2", R"("\x")", R"("\ud800")", "\"a\nb\""}) {
        EXPECT_THROW(Json::parse(bad), std::runtime_error) << bad;
    }
}

TEST(Json, AccessorsCheckKind) {
    const Json j = Json::parse(R"({"n": 1.5, "s": "x"})");
    EXPECT_THROW(j["n"].as_int(), std::runtime_error);
    EXPECT_THROW(j["s"].as_number(), std::runtime_error);
    EXPECT_THROW(j.as_array(), std::runtime_error);
}

} // namespace
