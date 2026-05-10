benchmark config: warmup=8 iters=50 train_warmup=1 train_iters=30
benchmark context: vocab=32 layers=4 d_model=256 d_ff=512 heads=8 kv_heads=4 head_dim=32 prefill_len=128 decode_prompt_len=127

=== Inference ===
-- Individual Kernels --
kernel: RMSNorm                                             60.81 us/iter
kernel: residual add                                         2.18 us/iter
kernel: Q projection                                        31.70 us/iter
kernel: RoPE rotation                                       12.66 us/iter
kernel: vocab softmax                                        0.17 us/iter
kernel: attention prefill                                  316.79 us/iter
kernel: attention decode                                     7.25 us/iter
kernel: MLP gate projection                                 50.56 us/iter
kernel: MLP up projection                                   49.65 us/iter
kernel: MLP SiLU gate multiply                              60.59 us/iter
kernel: MLP down projection                                 42.30 us/iter
-- Model Stages --
stage: token embedding                                       5.19 us/iter
stage: prefill whole prompt                               2912.11 us/iter
stage: decode first generated token                       3426.91 us/iter
stage: decode next generated token                         255.91 us/iter
stage: project hidden to vocab logits                        5.21 us/iter
stage: choose argmax token                                   0.58 us/iter
-- End-to-End Paths --
path: prefill then argmax last row                        3016.76 us/iter
path: prefill then decode one token                       3423.62 us/iter
path: decode 32 generated tokens                           192.91 us/token

=== Train ===
-- Individual Kernels --
kernel: grad final vocab projection                          1.94 us/iter
kernel: grad hidden from logits                              1.31 us/iter
kernel: grad token embedding                                 0.61 us/iter
kernel: linear layer backward                                8.94 us/iter
kernel: RMSNorm backward                                     7.05 us/iter
kernel: SiLU gate backward                                  19.38 us/iter
kernel: RoPE backward                                        0.73 us/iter
kernel: attention backward                                  46.99 us/iter
-- Training Step Breakdown --
step: reset gradient buffers                                88.62 us/iter
step: forward training pass                                750.96 us/iter
step: project selected logits                                1.82 us/iter
step: cross entropy loss                                     2.38 us/iter
step: grad final vocab projection                            1.90 us/iter
step: grad hidden from logits                                2.36 us/iter
step: transformer backward pass                           1138.19 us/iter
step: grad token embedding                                   0.95 us/iter
step: gradient clipping                                   3019.29 us/iter
step: AdamW optimizer update                              1159.42 us/iter
check: sum of listed train steps                          6165.88 us/iter
-- End-to-End Training --
full train_step()                                         6497.48 us/iter
full train_step() per token                                382.20 us/token
check: unlisted train_step overhead                        331.60 us/iter