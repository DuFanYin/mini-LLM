#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace io {

// Parsed JSON value; objects keep key order. Parse and access errors throw
// std::runtime_error.
class Json {
public:
    enum class Kind { null, boolean, number, string, array, object };

    static Json parse(std::string_view text);
    static Json parse_file(const std::string& path);

    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::null; }

    bool as_bool() const;
    double as_number() const;
    std::int64_t as_int() const; // throws unless the number is integral
    const std::string& as_string() const;
    const std::vector<Json>& as_array() const;
    const std::vector<std::pair<std::string, Json>>& as_object() const;

    const Json* find(std::string_view key) const; // nullptr if absent or not an object
    const Json& operator[](std::string_view key) const;

private:
    friend class JsonParser;

    Kind kind_ = Kind::null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Json> array_;
    std::vector<std::pair<std::string, Json>> object_;
};

} // namespace io
