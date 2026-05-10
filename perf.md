benchmark config: warmup=8 iters=50 train_warmup=1 train_iters=30
benchmark context: vocab=32 layers=4 d_model=256 d_ff=512 heads=8 kv_heads=4 head_dim=32 prefill_len=128 decode_prompt_len=127

=== Inference ===
-- Kernel Components --
inference.kernel.rms_norm                           33.50 us/iter
inference.kernel.add                                 4.07 us/iter
inference.kernel.linear_q                           23.33 us/iter
inference.kernel.apply_rope                          8.08 us/iter
inference.kernel.softmax_stable                      0.40 us/iter
inference.kernel.attention_prefill                4868.15 us/iter
inference.kernel.attention_decode                   40.97 us/iter
inference.kernel.mlp_gate_linear                    43.68 us/iter
inference.kernel.mlp_up_linear                      43.81 us/iter
inference.kernel.mlp_silu_mul                      113.35 us/iter
inference.kernel.mlp_down_linear                    38.86 us/iter
-- Pipeline Components --
inference.pipeline.embed                             4.31 us/iter
inference.pipeline.prefill                       24047.36 us/iter
inference.pipeline.decode_first_token            22639.14 us/iter
inference.pipeline.decode_steady_state             418.43 us/iter
inference.pipeline.project_logits                    5.20 us/iter
inference.pipeline.argmax_from_hidden_row            0.64 us/iter
-- End-to-End --
inference.e2e.last_argmax                        23625.49 us/iter
inference.e2e.first_token_argmax                 22021.49 us/iter
inference.e2e.decode_x32_tokens                    328.21 us/token

=== Train ===
-- Kernel Components --
train.kernel.backward_lm_head                        1.98 us/iter
train.kernel.backward_hidden                         1.28 us/iter
train.kernel.backward_embedding                      0.62 us/iter
train.kernel.linear_backward                         8.91 us/iter
train.kernel.rms_norm_backward                       6.86 us/iter
train.kernel.silu_mul_backward                      16.73 us/iter
train.kernel.apply_rope_backward                     0.73 us/iter
train.kernel.attention_prefill_backward             77.50 us/iter
-- Pipeline Components --
train.pipeline.forward_for_training                825.54 us/iter
train.pipeline.cross_entropy                         0.35 us/iter
train.pipeline.model_backward                     1338.33 us/iter
train.pipeline.adamw_update_model                 1252.78 us/iter
-- End-to-End --
train.e2e.train_step                              6873.42 us/iter
train.e2e.train_step_per_token                     518.74 us/token