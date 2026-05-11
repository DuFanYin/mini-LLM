#include "engine/io.h"
#include "model/mini_llm.h"
#include "task/task.h"
#include "train/train.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include <string>
#include <vector>

using namespace train;
using engine::save_model;

namespace {

void print_usage(const char* argv0) {
    std::println("usage: {} <max_steps> <log_every> <eval_every> <lr> <weight_decay> <max_grad_norm> "
                 "[save_path] [--seed rng]",
                 argv0);
}

size_t parse_size(const char* arg) {
    return static_cast<size_t>(std::strtoull(arg, nullptr, 10));
}

float parse_float(const char* arg) {
    return std::stof(arg);
}

[[nodiscard]] uint32_t parse_u32_seed(const char* arg) {
    return static_cast<uint32_t>(std::strtoul(arg, nullptr, 10));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        print_usage((argc > 0 && argv[0] != nullptr) ? argv[0] : "train");
        return 2;
    }

    const size_t max_steps = parse_size(argv[1]);
    const size_t log_every = parse_size(argv[2]);
    const size_t eval_every = parse_size(argv[3]);
    const float lr = parse_float(argv[4]);
    const float weight_decay = parse_float(argv[5]);
    const float max_grad_norm = parse_float(argv[6]);

    uint32_t model_seed = 42u;
    uint32_t train_sampler_seed = 2026u;
    uint32_t val_batch_seed = 2027u;
    uint32_t probe_seed = 9001u;

    std::string save_path;
    for (int i = 7; i < argc; ++i) {
        if (argv[i] == nullptr || argv[i][0] == '\0') {
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                print_usage((argc > 0 && argv[0] != nullptr) ? argv[0] : "train");
                return 2;
            }
            const uint32_t master = parse_u32_seed(argv[++i]);
            model_seed = master;
            train_sampler_seed = master + 1u;
            val_batch_seed = master + 2u;
            probe_seed = master + 3u;
            continue;
        }
        save_path = argv[i];
    }

    const auto run_t0 = std::chrono::steady_clock::now();
    auto elapsed_sec = [&run_t0]() -> double {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - run_t0).count();
    };

    constexpr size_t kVocab = task::n_vocab();
    std::unique_ptr<model::MiniLlm> model =
        std::make_unique<model::MiniLlm>(model::MiniLlm::init_random(/*vocab_size=*/kVocab, model_seed));
    model::ModelWeights adam_m = clone_model(model->weights());
    model::ModelWeights adam_v = clone_model(model->weights());

    model->configure_cache(128);

    task::Sampler sampler(train_sampler_seed);
    const auto val_set = task::batch_at_seed(/*count=*/256, val_batch_seed);

    float loss_sum = 0.0f;
    float acc_sum = 0.0f;
    double step_time_sum = 0.0;
    size_t counted = 0;

    constexpr float k_beta1 = 0.9f;
    constexpr float k_beta2 = 0.95f;
    constexpr float k_eps = 1e-8f;

    std::println("train loop (full-model AdamW + grad clip through every decoder block) | elapsed_s={:.3f}",
                 elapsed_sec());
    std::println(
        "task=which-span (PREFIX [SA] A [EA] MID [SB] B [EB] SUF [Q] -> copy chosen span), vocab={}, max_steps={}, "
        "log_every={}, eval_every={}, lr={}, wd={}, clip={}, "
        "seeds=model:{} train:{} val:{} probe:{} | elapsed_s={:.3f}",
        kVocab,
        max_steps,
        log_every,
        eval_every,
        lr,
        weight_decay,
        max_grad_norm,
        model_seed,
        train_sampler_seed,
        val_batch_seed,
        probe_seed,
        elapsed_sec());

    TrainWorkspace workspace{};
    bool reached_target = false;
    for (size_t step = 1; step <= max_steps; ++step) {
        const auto step_t0 = std::chrono::steady_clock::now();
        const auto token_ids = sampler.sample_sequence();
        task::answer_prediction_steps_into(token_ids, workspace.prediction_steps);
        const StepMetrics m =
            train_step(*model, token_ids, workspace.prediction_steps, step, lr, k_beta1, k_beta2, k_eps, weight_decay,
                       max_grad_norm, adam_m, adam_v, workspace);
        const double step_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - step_t0).count();

        loss_sum += m.loss;
        acc_sum += m.accuracy;
        step_time_sum += step_s;
        ++counted;

        if (step % log_every == 0) {
            std::println(
                "step {} | avg_loss={:.6f} | avg_acc={:.6f} | last_seq_len={} | "
                "last_step_s={:.3f} | avg_step_s={:.3f} | elapsed_s={:.3f}",
                         step,
                         loss_sum / static_cast<float>(counted),
                         acc_sum / static_cast<float>(counted),
                         token_ids.size(),
                         step_s,
                         step_time_sum / static_cast<double>(counted),
                         elapsed_sec());
            loss_sum = 0.0f;
            acc_sum = 0.0f;
            step_time_sum = 0.0;
            counted = 0;
        }

        if (step % eval_every == 0) {
            const auto eval_t0 = std::chrono::steady_clock::now();
            const float val_answer_acc = task::compute_answer_accuracy(*model, val_set);
            const auto [hits, trials] = task::count_probe_hits(*model, /*trials=*/16, probe_seed);
            const double eval_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_t0).count();
            const float probe_rate =
                (trials == 0) ? 0.0f : (static_cast<float>(hits) / static_cast<float>(trials));
            std::println("eval step {} | answer_acc={:.6f} | span_exact={}/{} | eval_s={:.3f} | elapsed_s={:.3f}",
                         step,
                         val_answer_acc,
                         hits,
                         trials,
                         eval_s,
                         elapsed_sec());
            if (val_answer_acc >= 0.995f && trials > 0 && probe_rate >= 0.995f) {
                reached_target = true;
                std::println("target reached at step {} | elapsed_s={:.3f}", step, elapsed_sec());
                break;
            }
        }
    }

    if (!reached_target) {
        std::println("done (target not reached within max_steps) | elapsed_s={:.3f}", elapsed_sec());
        if (!save_path.empty()) {
            save_model(save_path, model->config(), model->weights());
            std::println("model saved: {} | elapsed_s={:.3f}", save_path, elapsed_sec());
        }
        return 1;
    }
    if (!save_path.empty()) {
        save_model(save_path, model->config(), model->weights());
        std::println("model saved: {} | elapsed_s={:.3f}", save_path, elapsed_sec());
    }
    std::println("done (target reached) | elapsed_s={:.3f}", elapsed_sec());
    return 0;
}
