
**Summary**
| Method            |   accuracy |   precision |   recall |
|:-------------     |-----------:|------------:|---------:|
| Zero Shot         |   0.70784  |    0.83863  | 0.51472  |
| Fine-Tuned(256)   |   0.92360  |    0.92923  | 0.91704  |
| Fine-Tuned(LoRA)  |   0.91068  |    0.89946  | 0.92472  |


**Results for Zero Shot learning**

| bin          |   TP |   FP |   FN |   TN |   accuracy |   precision |   recall |
|:-------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------:|
| (0, 256]     | 4499 |  807 | 3252 | 6836 |   0.736326 |    0.847908 | 0.580441 |
| (256, 512]   | 1360 |  301 | 1865 | 3164 |   0.676233 |    0.818784 | 0.421705 |
| (512, 768]   |  354 |   78 |  611 |  820 |   0.630166 |    0.819444 | 0.366839 |
| (768, 1024]  |  137 |   39 |  215 |  286 |   0.624815 |    0.778409 | 0.389205 |
| (1024, 1280] |   73 |   10 |  100 |  150 |   0.66967  |    0.879518 | 0.421965 |

**Results for Fine Tuned model**
| bin          |   TP |   FP |   FN |   TN |   accuracy |   precision |   recall |
|:-------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------:|
| (0, 256]     | 7162 |  441 |  589 | 7202 |   0.933091 |    0.941997 | 0.92401  |
| (256, 512]   | 2936 |  276 |  289 | 3189 |   0.915546 |    0.914072 | 0.910388 |
| (512, 768]   |  873 |   95 |   92 |  803 |   0.899624 |    0.90186  | 0.904663 |
| (768, 1024]  |  308 |   45 |   44 |  280 |   0.868538 |    0.872521 | 0.875    |
| (1024, 1280] |  154 |   14 |   19 |  146 |   0.900901 |    0.916667 | 0.890173 |

**Results for Fine Tuned model with LoRA**
| bin          |   TP |   FP |   FN |   TN |   accuracy |   precision |   recall |
|:-------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------:|
| (0, 256]     | 7273 |  710 |  478 | 6933 |   0.922827 |    0.911061 | 0.938331 |
| (256, 512]   | 2919 |  376 |  306 | 3089 |   0.898057 |    0.885888 | 0.905116 |
| (512, 768]   |  866 |  119 |   99 |  779 |   0.882984 |    0.879188 | 0.897409 |
| (768, 1024]  |  314 |   63 |   38 |  262 |   0.850812 |    0.832891 | 0.892045 |
| (1024, 1280] |  156 |   20 |   17 |  140 |   0.888889 |    0.886364 | 0.901734 |
