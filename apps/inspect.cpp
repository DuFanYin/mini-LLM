// Loads a Qwen3.5 checkpoint onto the GPU and prints what was loaded.
//
//   inspect [checkpoint_dir]     (default: weights/qwen3.5-0.8b)

#include "core/device.h"
#include "io/safetensors.h"
#include "model/config.h"
#include "model/weights.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <print>
#include <string>

namespace {

constexpr double kMiB = 1024.0 * 1024.0;

double millions(std::size_t n) {
    return static_cast<double>(n) / 1e6;
}

void print_config(const model::Config& c) {
    std::println("model     hidden {} · mlp {} · vocab {} · eps {} · tied embedding {}", c.hidden_size,
                 c.intermediate_size, c.vocab_size, c.rms_norm_eps, c.tie_word_embeddings);
    std::string pattern;
    for (model::LayerKind k : c.layers) {
        pattern += k == model::LayerKind::full_attention ? 'A' : 'D';
    }
    std::println("layers    {} = {} GatedDeltaNet (D) + {} full attention (A): {}", c.num_layers(),
                 c.count(model::LayerKind::gated_delta_net), c.count(model::LayerKind::full_attention), pattern);
    std::println("attention heads {}/{} kv · head_dim {} · rotary {} · theta {:g}", c.attn.num_heads,
                 c.attn.num_kv_heads, c.attn.head_dim, c.attn.rotary_dim, c.attn.rope_theta);
    std::println("deltanet  heads {}k/{}v · k_dim {} · v_dim {} · conv {} over {} channels", c.gdn.num_k_heads,
                 c.gdn.num_v_heads, c.gdn.k_head_dim, c.gdn.v_head_dim, c.gdn.conv_kernel, c.gdn.conv_dim());
}

void print_weights(const model::Weights& w) {
    const auto& r = w.report();
    const auto part = [&](model::Part p) { return millions(r.params_by_part[static_cast<std::size_t>(p)]); };
    std::println("weights   {} tensors · {:.1f}M params · {:.1f} MiB on device · {:.2f} s", r.tensors,
                 millions(r.params), static_cast<double>(r.bytes) / kMiB, r.seconds);
    std::println("          embedding {:.1f}M · mlp {:.1f}M · deltanet {:.1f}M · attention {:.1f}M · norms {:.2f}M",
                 part(model::Part::embedding), part(model::Part::mlp), part(model::Part::gated_delta_net),
                 part(model::Part::full_attention), part(model::Part::norm));
    std::println("skipped   {} tensors · {:.1f}M params (vision encoder, MTP head)", r.skipped_tensors,
                 millions(r.skipped_params));
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : "weights/qwen3.5-0.8b";
    try {
        const core::DeviceInfo dev = core::current_device();
        std::println("device    {} · sm_{}{} · {} SMs · {:.0f}/{:.0f} MiB free", dev.name, dev.cc_major, dev.cc_minor,
                     dev.sm_count, static_cast<double>(dev.free_bytes) / kMiB,
                     static_cast<double>(dev.total_bytes) / kMiB);

        const model::Config config = model::Config::load(dir);
        print_config(config);

        const io::Checkpoint checkpoint(dir.string());
        const model::Weights weights = model::Weights::load(config, checkpoint);
        print_weights(weights);
    } catch (const std::exception& e) {
        std::println(stderr, "inspect: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
