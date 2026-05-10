benchmark config: warmup=8 iters=50 train_warmup=1 train_iters=30
benchmark context: vocab=32 layers=4 d_model=256 d_ff=512 heads=8 kv_heads=4 head_dim=32 prefill_len=128 decode_prompt_len=127

=== Inference ===
-- Individual Kernels --
kernel: RMSNorm                                             35.44 us/iter
kernel: residual add                                         1.47 us/iter
kernel: Q projection                                        21.74 us/iter
kernel: RoPE rotation                                        8.02 us/iter
kernel: vocab softmax                                        0.11 us/iter
kernel: attention prefill                                  249.03 us/iter
kernel: attention decode                                     5.49 us/iter
kernel: MLP gate projection                                 44.18 us/iter
kernel: MLP up projection                                   47.65 us/iter
kernel: MLP SiLU gate multiply                              55.14 us/iter
kernel: MLP down projection                                 39.99 us/iter
-- Model Stages --
stage: token embedding                                       3.44 us/iter
stage: prefill whole prompt                               2869.66 us/iter
stage: decode first generated token                       3344.08 us/iter
stage: decode next generated token                         244.24 us/iter
stage: project hidden to vocab logits                        5.29 us/iter
stage: choose argmax token                                   0.46 us/iter
-- End-to-End Paths --
path: prefill then argmax last row                        2991.74 us/iter
path: prefill then decode one token                       3429.18 us/iter
path: decode 32 generated tokens                           170.77 us/token

=== Train ===
-- Individual Kernels --
kernel: grad final vocab projection                          1.87 us/iter
kernel: grad hidden from logits                              1.28 us/iter
kernel: grad token embedding                                 0.59 us/iter
kernel: linear layer backward                                8.87 us/iter
kernel: RMSNorm backward                                     6.83 us/iter
kernel: SiLU gate backward                                  16.59 us/iter
kernel: RoPE backward                                        0.68 us/iter
kernel: attention backward                                  43.34 us/iter
-- Training Step Breakdown --
step: reset gradient buffers                                88.47 us/iter
step: forward training pass                                714.10 us/iter
step: project selected logits                                1.43 us/iter
step: cross entropy loss                                     1.71 us/iter
step: grad final vocab projection                            1.87 us/iter
step: grad hidden from logits                                2.04 us/iter
step: transformer backward pass                           1071.64 us/iter
step: grad token embedding                                   0.86 us/iter
step: gradient clipping                                    144.71 us/iter
step: AdamW optimizer update                               855.29 us/iter
check: sum of listed train steps                          2882.13 us/iter
-- End-to-End Training --
full train_step()                                         2905.41 us/iter
full train_step() per token                                170.91 us/token
check: unlisted train_step overhead                         23.28 us/iter