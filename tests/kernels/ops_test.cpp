// Each kernel against a direct CPU transcription of the same formula on small
// random inputs. These stay as the reference when kernels get optimised.

#include "core/allocator.h"
#include "core/stream.h"
#include "core/tensor.h"
#include "kernels/ops.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using core::DType;
using core::Tensor;

// Round-to-nearest-even f32 → bf16 bits, and back.
std::uint16_t to_bf16(float f) {
    const auto u = std::bit_cast<std::uint32_t>(f);
    return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(std::uint16_t b) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}
float silu(float x) {
    return x / (1.0f + std::exp(-x));
}
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

class Ops : public ::testing::Test {
protected:
    std::vector<float> random(std::size_t n, float lo = -1.0f, float hi = 1.0f) {
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> v(n);
        for (float& x : v) {
            x = d(rng_);
        }
        return v;
    }

    // Random weights, already rounded to bf16 so the CPU side sees the same values.
    std::vector<float> random_bf16(std::size_t n, float lo = -1.0f, float hi = 1.0f) {
        std::vector<float> v = random(n, lo, hi);
        for (float& x : v) {
            x = from_bf16(to_bf16(x));
        }
        return v;
    }

    Tensor f32(const std::vector<float>& v) {
        Tensor t({v.size()}, DType::f32, alloc_);
        t.copy_from_host(v.data());
        return t;
    }
    Tensor f32_zeros(std::size_t n) {
        Tensor t({n}, DType::f32, alloc_);
        t.zero();
        return t;
    }
    Tensor bf16(const std::vector<float>& v) {
        std::vector<std::uint16_t> bits(v.size());
        std::ranges::transform(v, bits.begin(), to_bf16);
        Tensor t({v.size()}, DType::bf16, alloc_);
        t.copy_from_host(bits.data());
        return t;
    }
    std::vector<float> host(const Tensor& t) {
        stream_.synchronize();
        std::vector<float> v(t.numel());
        t.copy_to_host(v.data());
        return v;
    }
    static const kernels::bf16* w(const Tensor& t) { return t.data<kernels::bf16>(); }

    static void expect_near(const std::vector<float>& got, const std::vector<float>& want, float tol = 1e-5f) {
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t i = 0; i < want.size(); ++i) {
            ASSERT_NEAR(got[i], want[i], tol * std::max(1.0f, std::abs(want[i]))) << "at " << i;
        }
    }

    std::mt19937 rng_{1234};
    core::PoolAllocator alloc_;
    core::Stream stream_;
};

TEST_F(Ops, RmsNormIsZeroCentred) {
    const std::size_t rows = 3, dim = 300;
    const auto x = random(rows * dim);
    const auto wv = random_bf16(dim, -0.5f, 0.5f);
    Tensor dx = f32(x), dw = bf16(wv), dy = f32_zeros(rows * dim);
    kernels::rms_norm(dx.data<float>(), rows, dim, w(dw), 1e-6f, dy.data<float>(), stream_.get());

    std::vector<float> want(rows * dim);
    for (std::size_t r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            ss += x[r * dim + i] * x[r * dim + i];
        }
        const float inv = 1.0f / std::sqrt(ss / dim + 1e-6f);
        for (std::size_t i = 0; i < dim; ++i) {
            want[r * dim + i] = x[r * dim + i] * inv * (1.0f + wv[i]);
        }
    }
    expect_near(host(dy), want);
}

TEST_F(Ops, LinearMultipliesByTransposedWeight) {
    const std::size_t rows = 3, in = 70, out = 45;
    const auto x = random(rows * in);
    const auto wv = random_bf16(out * in);
    Tensor dx = f32(x), dw = bf16(wv), dy = f32_zeros(rows * out);
    kernels::linear(dx.data<float>(), rows, in, w(dw), out, dy.data<float>(), stream_.get());

    std::vector<float> want(rows * out, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t n = 0; n < out; ++n) {
            for (std::size_t k = 0; k < in; ++k) {
                want[r * out + n] += x[r * in + k] * wv[n * in + k];
            }
        }
    }
    expect_near(host(dy), want);
}

TEST_F(Ops, HeadNormRopeRotatesLeadingDimsOfAStridedView) {
    const std::size_t tokens = 2, heads = 3, hd = 64, rot = 16, pos0 = 5;
    const double theta = 1e4;
    const std::size_t token_stride = heads * 2 * hd; // [query | gate] per head, like q_proj
    const auto x = random(tokens * token_stride);
    const auto wv = random_bf16(hd, -0.5f, 0.5f);
    Tensor dx = f32(x), dw = bf16(wv);
    kernels::head_norm_rope({dx.data<float>(), token_stride, 2 * hd}, tokens, heads, hd, w(dw), 1e-6f, rot, theta, pos0,
                            stream_.get());

    std::vector<float> want = x; // gate halves must stay untouched
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            float* v = want.data() + t * token_stride + h * 2 * hd;
            float ss = 0.0f;
            for (std::size_t d = 0; d < hd; ++d) {
                ss += v[d] * v[d];
            }
            const float inv = 1.0f / std::sqrt(ss / hd + 1e-6f);
            std::vector<float> n(hd);
            for (std::size_t d = 0; d < hd; ++d) {
                n[d] = v[d] * inv * (1.0f + wv[d]);
            }
            for (std::size_t d = 0; d < hd; ++d) {
                if (d >= rot) {
                    v[d] = n[d];
                    continue;
                }
                const std::size_t half = rot / 2;
                const double angle = static_cast<double>(pos0 + t) /
                                     std::pow(theta, 2.0 * static_cast<double>(d % half) / static_cast<double>(rot));
                const float rotated = d < half ? -n[d + half] : n[d - half];
                v[d] = n[d] * static_cast<float>(std::cos(angle)) + rotated * static_cast<float>(std::sin(angle));
            }
        }
    }
    expect_near(host(dx), want);
}

TEST_F(Ops, AttentionIsCausalOverTheCacheWithGroupedHeads) {
    const std::size_t tokens = 3, heads = 4, kvh = 2, hd = 32, pos0 = 2, max_seq = 8;
    const std::size_t total = pos0 + tokens;
    const auto k = random(total * kvh * hd);
    const auto v = random(total * kvh * hd);
    const auto q = random(tokens * heads * hd);
    Tensor dk = f32(k), dv = f32(v), dq = f32(q);
    Tensor kc = f32_zeros(kvh * max_seq * hd), vc = f32_zeros(kvh * max_seq * hd), out = f32_zeros(q.size());
    // Fill the cache in two appends: the past, then the current tokens.
    kernels::kv_append(dk.data<float>(), dv.data<float>(), kc.data<float>(), vc.data<float>(), pos0, kvh, hd, 0,
                       max_seq, stream_.get());
    kernels::kv_append(dk.data<float>() + pos0 * kvh * hd, dv.data<float>() + pos0 * kvh * hd, kc.data<float>(),
                       vc.data<float>(), tokens, kvh, hd, pos0, max_seq, stream_.get());
    kernels::attention({dq.data<float>(), heads * hd, hd}, kc.data<float>(), vc.data<float>(), max_seq,
                       out.data<float>(), tokens, heads, kvh, hd, pos0, stream_.get());

    std::vector<float> want(q.size(), 0.0f);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t g = h / (heads / kvh);
            const std::size_t visible = pos0 + t + 1;
            std::vector<float> p(visible);
            float m = -INFINITY;
            for (std::size_t l = 0; l < visible; ++l) {
                float dot = 0.0f;
                for (std::size_t d = 0; d < hd; ++d) {
                    dot += q[(t * heads + h) * hd + d] * k[(l * kvh + g) * hd + d];
                }
                p[l] = dot / std::sqrt(static_cast<float>(hd));
                m = std::max(m, p[l]);
            }
            float sum = 0.0f;
            for (float& x : p) {
                x = std::exp(x - m);
                sum += x;
            }
            for (std::size_t d = 0; d < hd; ++d) {
                for (std::size_t l = 0; l < visible; ++l) {
                    want[(t * heads + h) * hd + d] += p[l] / sum * v[(l * kvh + g) * hd + d];
                }
            }
        }
    }
    expect_near(host(out), want);
}

TEST_F(Ops, CausalConvCarriesStateAcrossCalls) {
    const std::size_t channels = 5, kernel = 4, tokens = 5;
    const auto x = random(tokens * channels);
    const auto wv = random_bf16(channels * kernel);
    Tensor dx = f32(x), dw = bf16(wv), state = f32_zeros((kernel - 1) * channels), out = f32_zeros(x.size());
    // Two calls (3 + 2 tokens) must equal one pass over all five.
    kernels::causal_conv1d_silu(dx.data<float>(), state.data<float>(), w(dw), 3, channels, kernel, out.data<float>(),
                                stream_.get());
    kernels::causal_conv1d_silu(dx.data<float>() + 3 * channels, state.data<float>(), w(dw), 2, channels, kernel,
                                out.data<float>() + 3 * channels, stream_.get());

    std::vector<float> want(x.size());
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < kernel; ++j) {
                const auto src = static_cast<std::ptrdiff_t>(t + j) - static_cast<std::ptrdiff_t>(kernel - 1);
                if (src >= 0) {
                    acc += x[static_cast<std::size_t>(src) * channels + c] * wv[c * kernel + j];
                }
            }
            want[t * channels + c] = silu(acc);
        }
    }
    expect_near(host(out), want);
    // The state holds the last kernel - 1 inputs.
    const auto s = host(state);
    for (std::size_t j = 0; j + 1 < kernel; ++j) {
        for (std::size_t c = 0; c < channels; ++c) {
            EXPECT_EQ(s[j * channels + c], x[(tokens - (kernel - 1) + j) * channels + c]);
        }
    }
}

TEST_F(Ops, GdnGatesFollowTheDiscretisation) {
    const std::size_t tokens = 4, heads = 3;
    const auto a = random(tokens * heads, -30.0f, 30.0f); // both softplus branches
    const auto b = random(tokens * heads, -5.0f, 5.0f);
    const auto a_log = random(heads);
    const auto dt = random_bf16(heads);
    Tensor da = f32(a), db = f32(b), dal = f32(a_log), ddt = bf16(dt), g = f32_zeros(a.size()),
           beta = f32_zeros(a.size());
    kernels::gdn_gates(da.data<float>(), db.data<float>(), dal.data<float>(), w(ddt), tokens, heads, g.data<float>(),
                       beta.data<float>(), stream_.get());

    std::vector<float> want_g(a.size()), want_beta(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float x = a[i] + dt[i % heads];
        want_g[i] = -std::exp(a_log[i % heads]) * static_cast<float>(std::log1p(std::exp(static_cast<double>(x))));
        want_beta[i] = sigmoid(b[i]);
    }
    expect_near(host(g), want_g);
    expect_near(host(beta), want_beta);
}

TEST_F(Ops, GdnRecurrenceMatchesTheDeltaRule) {
    const std::size_t heads = 2, kd = 32, vd = 32, tokens = 5;
    const std::size_t row = 2 * heads * kd + heads * vd;
    const auto qkv = random(tokens * row);
    const auto g = random(tokens * heads, -1.0f, 0.0f);
    const auto beta = random(tokens * heads, 0.0f, 1.0f);
    Tensor dqkv = f32(qkv), dg = f32(g), db = f32(beta), state = f32_zeros(heads * kd * vd),
           out = f32_zeros(tokens * heads * vd);
    // 3 + 2 tokens through the persistent state.
    kernels::gdn_recurrence(dqkv.data<float>(), dg.data<float>(), db.data<float>(), state.data<float>(),
                            out.data<float>(), 3, heads, heads, kd, vd, stream_.get());
    kernels::gdn_recurrence(dqkv.data<float>() + 3 * row, dg.data<float>() + 3 * heads, db.data<float>() + 3 * heads,
                            state.data<float>(), out.data<float>() + 3 * heads * vd, 2, heads, heads, kd, vd,
                            stream_.get());

    std::vector<float> want(tokens * heads * vd);
    std::vector<float> s(heads * kd * vd, 0.0f);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            std::vector<float> q(kd), k(kd), v(vd);
            float qn = 0.0f, kn = 0.0f;
            for (std::size_t i = 0; i < kd; ++i) {
                q[i] = qkv[t * row + h * kd + i];
                k[i] = qkv[t * row + heads * kd + h * kd + i];
                qn += q[i] * q[i];
                kn += k[i] * k[i];
            }
            for (std::size_t i = 0; i < kd; ++i) {
                q[i] = q[i] / std::sqrt(qn + 1e-6f) / std::sqrt(static_cast<float>(kd));
                k[i] = k[i] / std::sqrt(kn + 1e-6f);
            }
            for (std::size_t j = 0; j < vd; ++j) {
                v[j] = qkv[t * row + 2 * heads * kd + h * vd + j];
            }
            float* sh = s.data() + h * kd * vd;
            const float decay = std::exp(g[t * heads + h]);
            for (std::size_t j = 0; j < vd; ++j) {
                float mem = 0.0f;
                for (std::size_t i = 0; i < kd; ++i) {
                    sh[i * vd + j] *= decay;
                    mem += sh[i * vd + j] * k[i];
                }
                const float delta = (v[j] - mem) * beta[t * heads + h];
                float o = 0.0f;
                for (std::size_t i = 0; i < kd; ++i) {
                    sh[i * vd + j] += k[i] * delta;
                    o += sh[i * vd + j] * q[i];
                }
                want[(t * heads + h) * vd + j] = o;
            }
        }
    }
    expect_near(host(out), want);
    expect_near(host(state), s);
}

TEST_F(Ops, GatedRmsNormUsesPlainWeightAndSiluGate) {
    const std::size_t rows = 4, dim = 128;
    const auto x = random(rows * dim);
    const auto z = random(rows * dim, -3.0f, 3.0f);
    const auto wv = random(dim);
    Tensor dx = f32(x), dz = f32(z), dw = f32(wv), out = f32_zeros(x.size());
    kernels::gated_rms_norm(dx.data<float>(), dz.data<float>(), dw.data<float>(), 1e-6f, rows, dim, out.data<float>(),
                            stream_.get());

    std::vector<float> want(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            ss += x[r * dim + i] * x[r * dim + i];
        }
        const float inv = 1.0f / std::sqrt(ss / dim + 1e-6f);
        for (std::size_t i = 0; i < dim; ++i) {
            want[r * dim + i] = x[r * dim + i] * inv * wv[i] * silu(z[r * dim + i]);
        }
    }
    expect_near(host(out), want);
}

TEST_F(Ops, ArgmaxPrefersTheLowestIndexOnTies) {
    std::vector<float> x = random(300'000);
    x[123'456] = 5.0f;
    x[200'000] = 5.0f;
    x[299'999] = 5.0f;
    Tensor dx = f32(x);
    Tensor out({1}, DType::i32, alloc_);
    kernels::argmax(dx.data<float>(), x.size(), out.data<std::int32_t>(), stream_.get());
    stream_.synchronize();
    std::int32_t got = -1;
    out.copy_to_host(&got);
    EXPECT_EQ(got, 123'456);
}

TEST_F(Ops, EmbeddingGathersRows) {
    const std::size_t vocab = 10, dim = 7;
    const auto table = random_bf16(vocab * dim);
    const std::vector<std::int32_t> ids = {3, 0, 9, 3};
    Tensor dt = bf16(table), out = f32_zeros(ids.size() * dim);
    Tensor did({ids.size()}, DType::i32, alloc_);
    did.copy_from_host(ids.data());
    kernels::embedding(did.data<std::int32_t>(), ids.size(), w(dt), dim, out.data<float>(), stream_.get());

    std::vector<float> want;
    for (std::int32_t id : ids) {
        want.insert(want.end(), table.begin() + id * static_cast<std::ptrdiff_t>(dim),
                    table.begin() + (id + 1) * static_cast<std::ptrdiff_t>(dim));
    }
    EXPECT_EQ(host(out), want);
}

} // namespace
