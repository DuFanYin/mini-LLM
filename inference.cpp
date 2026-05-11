// Load a trained model for Which-Span task and run interactive retrieval-copy inference.

#include "model/mini_llm.h"
#include "engine/decode.h"
#include "task/task.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <span>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::println("usage: {} [model.ckpt] [--temperature T] [--seed N]", argv0);
    std::println("");
    std::println("Input: 6 A-Z words separated by spaces:");
    std::println("    PREFIX  SPAN_A  MIDDLE  SPAN_B  SUFFIX  Q");
    std::println("Q is A or B; the model should print SPAN_A or SPAN_B.");
    std::println("");
    std::println("Examples:");
    std::println("    AB CD EF GH IJ A   -> CD");
    std::println("    AB CD EF GH IJ B   -> GH");
    std::println("    HELLO CAT MID DOG ZZ B   -> DOG");
    std::println("");
    std::println("Type exit or quit to leave.");
}

[[nodiscard]] std::string trim_copy(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

[[nodiscard]] bool parse_letters_field(const std::string& word, size_t min_len, size_t max_len,
                                     std::vector<uint32_t>* out, std::string* err, const char* segment_name) {
    if (word.size() < min_len || word.size() > max_len) {
        *err = std::string(segment_name) + " length must be in [" + std::to_string(min_len) + ".." +
               std::to_string(max_len) + "]";
        return false;
    }
    out->clear();
    out->reserve(word.size());
    for (unsigned char uc : word) {
        const char ch = static_cast<char>(std::toupper(uc));
        if (ch < 'A' || ch > 'Z') {
            *err = "letters must be A-Z only";
            return false;
        }
        out->push_back(static_cast<uint32_t>(ch - 'A'));
    }
    return true;
}

[[nodiscard]] bool parse_input(const std::string& raw, std::vector<uint32_t>* prompt_through_query,
                               std::vector<uint32_t>* gold_span, std::string* err) {
    std::istringstream iss(raw);
    std::vector<std::string> parts;
    std::string w;
    while (iss >> w) {
        parts.push_back(std::move(w));
    }
    if (parts.size() != 6) {
        *err = "expected 6 fields: PREFIX SPAN_A MIDDLE SPAN_B SUFFIX Q (e.g. AB CD EF GH IJ A)";
        return false;
    }

    std::vector<uint32_t> prefix;
    std::vector<uint32_t> span_a;
    std::vector<uint32_t> middle;
    std::vector<uint32_t> span_b;
    std::vector<uint32_t> suffix;

    if (!parse_letters_field(parts[0], task::prefix_len_min(), task::prefix_len_max(), &prefix, err, "prefix")) {
        return false;
    }
    if (!parse_letters_field(parts[1], task::span_a_len_min(), task::span_a_len_max(), &span_a, err, "span_a")) {
        return false;
    }
    if (!parse_letters_field(parts[2], task::middle_len_min(), task::middle_len_max(), &middle, err, "middle")) {
        return false;
    }
    if (!parse_letters_field(parts[3], task::span_b_len_min(), task::span_b_len_max(), &span_b, err, "span_b")) {
        return false;
    }
    if (!parse_letters_field(parts[4], task::suffix_len_min(), task::suffix_len_max(), &suffix, err, "suffix")) {
        return false;
    }

    if (parts[5].size() != 1) {
        *err = "Q must be A or B";
        return false;
    }
    const char qch = static_cast<char>(std::toupper(static_cast<unsigned char>(parts[5][0])));
    const bool query_a = (qch == 'A');
    const bool query_b = (qch == 'B');
    if (!query_a && !query_b) {
        *err = "Q must be A or B";
        return false;
    }

    prompt_through_query->clear();
    prompt_through_query->reserve(prefix.size() + 1u + span_a.size() + 1u + middle.size() + 1u + span_b.size() + 1u +
                                  suffix.size() + 1u);
    prompt_through_query->insert(prompt_through_query->end(), prefix.begin(), prefix.end());
    prompt_through_query->push_back(task::tok_sa());
    prompt_through_query->insert(prompt_through_query->end(), span_a.begin(), span_a.end());
    prompt_through_query->push_back(task::tok_ea());
    prompt_through_query->insert(prompt_through_query->end(), middle.begin(), middle.end());
    prompt_through_query->push_back(task::tok_sb());
    prompt_through_query->insert(prompt_through_query->end(), span_b.begin(), span_b.end());
    prompt_through_query->push_back(task::tok_eb());
    prompt_through_query->insert(prompt_through_query->end(), suffix.begin(), suffix.end());
    prompt_through_query->push_back(query_a ? task::tok_qa() : task::tok_qb());

    *gold_span = query_a ? span_a : span_b;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    print_usage((argc > 0 && argv[0] != nullptr) ? argv[0] : "inference");

    std::string model_path = "weights/model.ckpt";
    float temperature = 1.0f;
    bool seed_provided = false;
    uint32_t sample_seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == nullptr || argv[i][0] == '\0') continue;
        if (std::strcmp(argv[i], "--temperature") == 0 || std::strcmp(argv[i], "--temp") == 0) {
            if (i + 1 < argc && argv[++i][0] != '\0') {
                char* end = nullptr;
                const float v = std::strtof(argv[i], &end);
                if (end != argv[i]) {
                    temperature = v;
                }
            }
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc && argv[++i][0] != '\0') {
                sample_seed = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10));
                seed_provided = true;
            }
            continue;
        }
        if (argv[i][0] == '-') {
            std::println("unknown option: {}", argv[i]);
            return 2;
        }
        model_path = argv[i];
    }

    std::mt19937 rng;
    if (seed_provided) {
        rng.seed(sample_seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    std::unique_ptr<model::MiniLlm> model = model::MiniLlm::init_load(model_path, 256);

    if (model->weights().vocab_size < task::n_vocab()) {
        std::println("error: model vocab_size={} is smaller than task vocab ({})",
                     model->weights().vocab_size,
                     task::n_vocab());
        return 1;
    }

    std::println("");
    std::println("loaded: {}", model_path);
    if (seed_provided) {
        std::println("decode sampling: temperature={}, seed={}", temperature, sample_seed);
    } else {
        std::println("decode sampling: temperature={}, seed=(random per launch)", temperature);
    }
    std::println("");

    std::string line;
    while (true) {
        std::print("> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line == "exit" || line == "quit") break;

        std::vector<uint32_t> prompt;
        std::vector<uint32_t> gold_span;
        std::string err;
        if (!parse_input(line, &prompt, &gold_span, &err)) {
            std::println("invalid: {}", err);
            continue;
        }

        const model::ModelWeights& weights = model->weights();
        const size_t d_model = model->d_model();
        const size_t vocab_size = weights.vocab_size;

        // Prefill: feed the whole prompt once, fill KV cache, get hidden states for every prompt position.
        std::vector<float> prompt_hidden(prompt.size() * d_model, 0.0f);
        model->prefill(prompt, prompt_hidden.data());

        // First answer token comes from the last prompt position's hidden row.
        std::vector<uint32_t> pred_span;
        pred_span.reserve(gold_span.size());
        const float* last_row = prompt_hidden.data() + (prompt.size() - 1u) * d_model;
        uint32_t token = engine::sample_from_hidden_row(std::span<const float>(last_row, d_model), weights.output_projection,
                                                        vocab_size, d_model, temperature, rng);
        pred_span.push_back(token);

        // Decode: one token in, one hidden row out, KV cache reused. Repeat for the rest of the answer span.
        std::vector<float> decode_row(d_model, 0.0f);
        for (size_t i = 1; i < gold_span.size(); ++i) {
            model->decode(token, decode_row.data());
            token = engine::sample_from_hidden_row(std::span<const float>(decode_row.data(), d_model), weights.output_projection,
                                                   vocab_size, d_model, temperature, rng);
            pred_span.push_back(token);
        }

        auto to_letters = [](const std::vector<uint32_t>& ids) {
            std::string s;
            s.reserve(ids.size());
            for (uint32_t id : ids) {
                s.push_back(task::to_char(id));
            }
            return s;
        };
        const bool exact = pred_span == gold_span;
        std::println("target={} | pred={} | exact={}", to_letters(gold_span), to_letters(pred_span), exact ? "yes" : "no");
    }
    return 0;
}
