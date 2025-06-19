# Garbled-GPT2 Benchmark Results

**Total Runs:** 5

**Input Size:** 768 elements

## Garbler Performance

| Metric | Mean | Median | Min | Max | Std Dev |
|--------|------|--------|-----|-----|---------|
| Wall Time (ms) | 43.62 | 39.61 | 36.98 | 56.79 | 8.07 |
| Garble Time (ms) | 43.49 | 39.07 | 36.95 | 56.76 | 8.13 |
| Eval Time (ms) | 0.70 | 0.69 | 0.64 | 0.76 | 0.04 |
| Reveal Time (ms) | 2.67 | 2.65 | 2.10 | 3.31 | 0.44 |
| Bytes Sent | 91152.00 | 91152.00 | 91152.00 | 91152.00 | 0.00 |
| Bytes Received | 4142.00 | 4142.00 | 4142.00 | 4142.00 | 0.00 |
| Total Bytes | 95294.00 | 95294.00 | 95294.00 | 95294.00 | 0.00 |
| Peak Memory (MB) | 0.68 | 0.66 | 0.65 | 0.78 | 0.05 |
| Memory Delta (MB) | 0.49 | 0.55 | 0.36 | 0.56 | 0.09 |

## Evaluator Performance

| Metric | Mean | Median | Min | Max | Std Dev |
|--------|------|--------|-----|-----|---------|
| Wall Time (ms) | 43.72 | 39.81 | 37.00 | 57.02 | 8.13 |
| Eval Time (ms) | 0.67 | 0.65 | 0.62 | 0.72 | 0.04 |
| Reveal Time (ms) | 0.79 | 0.84 | 0.52 | 1.05 | 0.20 |
| Bytes Sent | 4142.00 | 4142.00 | 4142.00 | 4142.00 | 0.00 |
| Bytes Received | 91152.00 | 91152.00 | 91152.00 | 91152.00 | 0.00 |
| Total Bytes | 95294.00 | 95294.00 | 95294.00 | 95294.00 | 0.00 |
| Peak Memory (MB) | 0.74 | 0.74 | 0.74 | 0.75 | 0.00 |
| Memory Delta (MB) | 0.56 | 0.60 | 0.42 | 0.60 | 0.08 |

## Combined Overview

**Average Garbler Time:** 43.62 ms
**Average Evaluator Time:** 43.72 ms
**Average Communication:** 95294 bytes (93.1 KB)
