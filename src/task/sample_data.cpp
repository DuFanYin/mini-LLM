#include "task/task.h"

#include <random>
#include <vector>

namespace task {

namespace {

constexpr size_t kNone = static_cast<size_t>(-1);

} // namespace

Sampler::Sampler(uint32_t seed) : rng_(seed) {}

std::vector<uint32_t> Sampler::sample_sequence() {
    std::uniform_int_distribution<uint32_t> letter_dist(0u, 25u);
    std::uniform_int_distribution<size_t> prefix_dist(prefix_len_min(), prefix_len_max());
    std::uniform_int_distribution<size_t> middle_dist(middle_len_min(), middle_len_max());
    std::uniform_int_distribution<size_t> suffix_dist(suffix_len_min(), suffix_len_max());
    std::uniform_int_distribution<size_t> span_a_dist(span_a_len_min(), span_a_len_max());
    std::uniform_int_distribution<size_t> span_b_dist(span_b_len_min(), span_b_len_max());
    std::bernoulli_distribution query_pick_a(0.5);

    const size_t prefix_n = prefix_dist(rng_);
    const size_t middle_n = middle_dist(rng_);
    const size_t suffix_n = suffix_dist(rng_);
    const size_t span_a_n = span_a_dist(rng_);
    const size_t span_b_n = span_b_dist(rng_);
    const bool use_qa = query_pick_a(rng_);

    std::vector<uint32_t> span_a;
    span_a.reserve(span_a_n);
    std::vector<uint32_t> span_b;
    span_b.reserve(span_b_n);
    for (size_t i = 0; i < span_a_n; ++i) {
        span_a.push_back(letter_dist(rng_));
    }
    for (size_t i = 0; i < span_b_n; ++i) {
        span_b.push_back(letter_dist(rng_));
    }

    std::vector<uint32_t> out;
    out.reserve(prefix_n + 1u + span_a_n + 1u + middle_n + 1u + span_b_n + 1u + suffix_n + 1u +
                (use_qa ? span_a_n : span_b_n));

    for (size_t i = 0; i < prefix_n; ++i) {
        out.push_back(letter_dist(rng_));
    }
    out.push_back(tok_sa());
    out.insert(out.end(), span_a.begin(), span_a.end());
    out.push_back(tok_ea());
    for (size_t i = 0; i < middle_n; ++i) {
        out.push_back(letter_dist(rng_));
    }
    out.push_back(tok_sb());
    out.insert(out.end(), span_b.begin(), span_b.end());
    out.push_back(tok_eb());
    for (size_t i = 0; i < suffix_n; ++i) {
        out.push_back(letter_dist(rng_));
    }
    out.push_back(use_qa ? tok_qa() : tok_qb());
    if (use_qa) {
        out.insert(out.end(), span_a.begin(), span_a.end());
    } else {
        out.insert(out.end(), span_b.begin(), span_b.end());
    }
    return out;
}

std::vector<std::vector<uint32_t>> Sampler::sample_batch(size_t batch_size) {
    std::vector<std::vector<uint32_t>> batch;
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(sample_sequence());
    }
    return batch;
}

bool infer_layout(const std::vector<uint32_t>& t, Layout& out) noexcept {
    size_t isa = kNone;
    size_t iea = kNone;
    size_t isb = kNone;
    size_t ieb = kNone;
    size_t iqa = kNone;
    size_t iqb = kNone;
    for (size_t i = 0; i < t.size(); ++i) {
        const uint32_t x = t[i];
        if (x == tok_sa()) {
            if (isa != kNone) {
                return false;
            }
            isa = i;
        } else if (x == tok_ea()) {
            if (iea != kNone) {
                return false;
            }
            iea = i;
        } else if (x == tok_sb()) {
            if (isb != kNone) {
                return false;
            }
            isb = i;
        } else if (x == tok_eb()) {
            if (ieb != kNone) {
                return false;
            }
            ieb = i;
        } else if (x == tok_qa()) {
            if (iqa != kNone) {
                return false;
            }
            iqa = i;
        } else if (x == tok_qb()) {
            if (iqb != kNone) {
                return false;
            }
            iqb = i;
        }
    }
    if (isa == kNone || iea == kNone || isb == kNone || ieb == kNone) {
        return false;
    }
    const bool has_qa = iqa != kNone;
    const bool has_qb = iqb != kNone;
    if (has_qa == has_qb) {
        return false;
    }
    const size_t qidx = has_qa ? iqa : iqb;
    if (!(isa < iea && iea < isb && isb < ieb && ieb < qidx)) {
        return false;
    }

    const size_t prefix_n = isa;
    const size_t span_a_n = iea - isa - 1u;
    const size_t middle_n = isb - iea - 1u;
    const size_t span_b_n = ieb - isb - 1u;
    const size_t suffix_n = qidx - ieb - 1u;

    if (prefix_n < prefix_len_min() || prefix_n > prefix_len_max()) {
        return false;
    }
    if (middle_n < middle_len_min() || middle_n > middle_len_max()) {
        return false;
    }
    if (suffix_n < suffix_len_min() || suffix_n > suffix_len_max()) {
        return false;
    }
    if (span_a_n < span_a_len_min() || span_a_n > span_a_len_max()) {
        return false;
    }
    if (span_b_n < span_b_len_min() || span_b_n > span_b_len_max()) {
        return false;
    }

    out.idx_sa = isa;
    out.idx_ea = iea;
    out.idx_sb = isb;
    out.idx_eb = ieb;
    out.idx_query = qidx;
    out.query_is_a = has_qa;
    out.answer_start = qidx + 1u;
    out.answer_len = has_qa ? span_a_n : span_b_n;

    if (t.size() != qidx + 1u + out.answer_len) {
        return false;
    }

    return true;
}

bool is_valid(const std::vector<uint32_t>& t) noexcept {
    Layout layout;
    if (!infer_layout(t, layout)) {
        return false;
    }
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] >= n_vocab()) {
            return false;
        }
    }
    const size_t gold_begin = layout.query_is_a ? (layout.idx_sa + 1u) : (layout.idx_sb + 1u);
    for (size_t k = 0; k < layout.answer_len; ++k) {
        if (t[layout.answer_start + k] != t[gold_begin + k]) {
            return false;
        }
    }
    return true;
}

void answer_prediction_steps_into(const std::vector<uint32_t>& t, std::vector<size_t>& out) {
    out.clear();
    Layout layout;
    if (!infer_layout(t, layout)) {
        return;
    }
    out.reserve(layout.answer_len);
    for (size_t i = 0; i < layout.answer_len; ++i) {
        out.push_back(layout.idx_query + i);
    }
}

std::vector<size_t> answer_prediction_steps(const std::vector<uint32_t>& t) {
    std::vector<size_t> steps;
    answer_prediction_steps_into(t, steps);
    return steps;
}

std::vector<std::vector<uint32_t>> batch_at_seed(size_t count, uint32_t seed) {
    Sampler sampler(seed);
    return sampler.sample_batch(count);
}

} // namespace task
