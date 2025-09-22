### Top 5 Generative
| F1 | EM | NoAns | AnsF1 | p50_ms | top_k | max_distance | null_threshold | rerank | rerank_lex_w | alpha | alpha_hits | support_min | support_window | span_max_distance | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.542 | 0.518 | 0.856 | 0.194 | 127 | 3 | 0.6 | 0.23 | False | 0.5 |  |  |  |  |  | gen_k3_md0p60_nt0p23_rr0.json |
| 0.533 | 0.522 | 0.916 | 0.108 | 100 | 3 | 0.6 | 0.2 | True | 0.5 |  |  |  |  |  | gen_k3_md0p60_nt0p20_rr1_rw0p5.json |
| 0.532 | 0.512 | 0.84 | 0.19 | 155 | 3 | 0.6 | 0.23 | True | 0.5 |  |  |  |  |  | gen_k3_md0p60_nt0p23_rr1_rw0p5.json |
| 0.532 | 0.51 | 0.852 | 0.176 | 133 | 5 | 0.6 | 0.23 | False | 0.5 |  |  |  |  |  | gen_k5_md0p60_nt0p23_rr0.json |
| 0.531 | 0.522 | 0.92 | 0.1 | 118 | 5 | 0.6 | 0.2 | True | 0.5 |  |  |  |  |  | gen_k5_md0p60_nt0p20_rr1_rw0p5.json |

### Top 5 Extractive
| F1 | EM | NoAns | AnsF1 | p50_ms | top_k | max_distance | null_threshold | rerank | rerank_lex_w | alpha | alpha_hits | support_min | support_window | span_max_distance | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.533 | 0.49 | 0.669 | 0.382 | 1448 | 3 | 0.6 | 0.6 | True | 0.5 | 0.45 | 2.0 | 0.3 | 96.0 | 0.6 | ex_k3_md0p60_nt0p60_rr1_rw0p5_a0p45_h2_sm0p30_w96_sd0p60.json |
| 0.532 | 0.492 | 0.669 | 0.381 | 1342 | 3 | 0.6 | 0.6 | True | 0.5 | 0.55 | 1.0 | 0.3 | 96.0 | 0.5 | ex_k3_md0p60_nt0p60_rr1_rw0p5_a0p55_h1_sm0p30_w96_sd0p50.json |
| 0.53 | 0.488 | 0.665 | 0.381 | 1303 | 3 | 0.6 | 0.55 | True | 0.5 | 0.5 | 2.0 | 0.3 | 96.0 | 0.5 | ex_k3_md0p60_nt0p55_rr1_rw0p5_a0p50_h2_sm0p30_w96_sd0p50.json |
| 0.53 | 0.488 | 0.669 | 0.375 | 1429 | 3 | 0.6 | 0.65 | True | 0.5 | 0.5 | 2.0 | 0.3 | 96.0 | 0.5 | ex_k3_md0p60_nt0p65_rr1_rw0p5_a0p50_h2_sm0p30_w96_sd0p50.json |
| 0.53 | 0.49 | 0.669 | 0.375 | 1297 | 3 | 0.6 | 0.6 | True | 0.5 | 0.55 | 2.0 | 0.3 | 96.0 | 0.6 | ex_k3_md0p60_nt0p60_rr1_rw0p5_a0p55_h2_sm0p30_w96_sd0p60.json |