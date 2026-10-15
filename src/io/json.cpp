#include "io/json.h"

#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace io {

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : s_(text) {}

    Json parse_document() {
        Json v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) {
            fail("trailing characters");
        }
        return v;
    }

private:
    static constexpr int kMaxDepth = 256;

    [[noreturn]] void fail(std::string_view what) const {
        throw std::runtime_error(std::format("json: {} at offset {}", what, pos_));
    }

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    char peek() {
        skip_ws();
        if (pos_ >= s_.size()) {
            fail("unexpected end");
        }
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::format("expected '{}'", c));
        }
        ++pos_;
    }

    void expect_word(std::string_view w) {
        if (s_.substr(pos_, w.size()) != w) {
            fail("invalid literal");
        }
        pos_ += w.size();
    }

    Json parse_value(int depth) {
        if (depth > kMaxDepth) {
            fail("nesting too deep");
        }
        Json v;
        switch (peek()) {
        case '{':
            v.kind_ = Json::Kind::object;
            parse_object(v, depth);
            break;
        case '[':
            v.kind_ = Json::Kind::array;
            parse_array(v, depth);
            break;
        case '"':
            v.kind_ = Json::Kind::string;
            v.string_ = parse_string();
            break;
        case 't':
            expect_word("true");
            v.kind_ = Json::Kind::boolean;
            v.bool_ = true;
            break;
        case 'f':
            expect_word("false");
            v.kind_ = Json::Kind::boolean;
            break;
        case 'n':
            expect_word("null");
            break;
        default:
            v.kind_ = Json::Kind::number;
            v.number_ = parse_number();
        }
        return v;
    }

    void parse_object(Json& v, int depth) {
        expect('{');
        if (peek() == '}') {
            ++pos_;
            return;
        }
        while (true) {
            if (peek() != '"') {
                fail("expected key");
            }
            std::string key = parse_string();
            expect(':');
            v.object_.emplace_back(std::move(key), parse_value(depth + 1));
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return;
        }
    }

    void parse_array(Json& v, int depth) {
        expect('[');
        if (peek() == ']') {
            ++pos_;
            return;
        }
        while (true) {
            v.array_.push_back(parse_value(depth + 1));
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return;
        }
    }

    double parse_number() {
        const char* begin = s_.data() + pos_;
        const char* end = s_.data() + s_.size();
        double out = 0.0;
        auto [ptr, ec] = std::from_chars(begin, end, out);
        if (ec != std::errc{} || ptr == begin) {
            fail("invalid number");
        }
        pos_ += static_cast<std::size_t>(ptr - begin);
        return out;
    }

    unsigned hex4() {
        if (pos_ + 4 > s_.size()) {
            fail("short \\u escape");
        }
        unsigned cp = 0;
        auto [ptr, ec] = std::from_chars(s_.data() + pos_, s_.data() + pos_ + 4, cp, 16);
        if (ec != std::errc{} || ptr != s_.data() + pos_ + 4) {
            fail("invalid \\u escape");
        }
        pos_ += 4;
        return cp;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parse_string() {
        ++pos_; // opening quote
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) {
                fail("unterminated string");
            }
            const char c = s_[pos_++];
            if (c == '"') {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("control character in string");
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) {
                fail("unterminated escape");
            }
            const char e = s_[pos_++];
            switch (e) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xD800 && cp < 0xDC00) {
                    if (s_.substr(pos_, 2) != "\\u") {
                        fail("lone high surrogate");
                    }
                    pos_ += 2;
                    const unsigned lo = hex4();
                    if (lo < 0xDC00 || lo >= 0xE000) {
                        fail("invalid low surrogate");
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                } else if (cp >= 0xDC00 && cp < 0xE000) {
                    fail("lone low surrogate");
                }
                append_utf8(out, cp);
                break;
            }
            default:
                fail("invalid escape");
            }
        }
    }

    std::string_view s_;
    std::size_t pos_ = 0;
};

Json Json::parse(std::string_view text) {
    return JsonParser(text).parse_document();
}

Json Json::parse_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::format("json: cannot open {}", path));
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    try {
        return parse(buf.str());
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::format("{}: {}", path, e.what()));
    }
}

namespace {

[[noreturn]] void wrong_kind(std::string_view want) {
    throw std::runtime_error(std::format("json: value is not {}", want));
}

} // namespace

bool Json::as_bool() const {
    if (kind_ != Kind::boolean) {
        wrong_kind("a boolean");
    }
    return bool_;
}

double Json::as_number() const {
    if (kind_ != Kind::number) {
        wrong_kind("a number");
    }
    return number_;
}

std::int64_t Json::as_int() const {
    const double n = as_number();
    if (std::trunc(n) != n || std::fabs(n) > 9007199254740992.0) {
        wrong_kind("an integer");
    }
    return static_cast<std::int64_t>(n);
}

const std::string& Json::as_string() const {
    if (kind_ != Kind::string) {
        wrong_kind("a string");
    }
    return string_;
}

const std::vector<Json>& Json::as_array() const {
    if (kind_ != Kind::array) {
        wrong_kind("an array");
    }
    return array_;
}

const std::vector<std::pair<std::string, Json>>& Json::as_object() const {
    if (kind_ != Kind::object) {
        wrong_kind("an object");
    }
    return object_;
}

const Json* Json::find(std::string_view key) const {
    if (kind_ != Kind::object) {
        return nullptr;
    }
    for (const auto& [k, v] : object_) {
        if (k == key) {
            return &v;
        }
    }
    return nullptr;
}

const Json& Json::operator[](std::string_view key) const {
    as_object();
    const Json* v = find(key);
    if (v == nullptr) {
        throw std::runtime_error(std::format("json: missing key '{}'", key));
    }
    return *v;
}

} // namespace io
