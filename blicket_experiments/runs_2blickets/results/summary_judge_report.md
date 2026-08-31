# Summary judge report

Models found: 22

## obj_4 / disjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 11.00 | 0.750 | 0.681 | 0.155 | 75 | 57 | 156 | 24 |
| llama-3.3-70b-instruct | 8.44 | 1.000 | 3.366 |  | 43 | 92 | 0 | 0 |
| qwen3-14b | 6.47 | 0.938 | 0.968 | 0.184 | 49 | 48 | 38 | 7 |
| qwen3-14b-nothink | 11.75 | 0.250 | 0.895 | 0.581 | 26 | 21 | 364 | 176 |
| qwen3-32b | 7.38 | 1.000 | 1.931 |  | 52 | 66 | 0 | 0 |
| qwen3-32b-nothink | 10.40 | 0.938 | 1.554 | 1.368 | 67 | 89 | 19 | 26 |
| qwen3-32bstudent-32btioteacher | 5.50 | 1.000 | 2.613 |  | 36 | 52 | 0 | 0 |
| qwen3-8b | 15.40 | 0.312 | 1.150 | 1.805 | 34 | 42 | 234 | 261 |
| qwen3-8b-nothink | 17.00 | 0.250 | 0.702 | 1.781 | 39 | 29 | 257 | 283 |
| qwen3-8bstudent-32bbaselineteacher | 7.83 | 0.750 | 0.761 | 0.234 | 69 | 25 | 147 | 33 |
| qwen3-8bstudent-32btioteacher | 10.77 | 0.812 | 0.555 | 0.064 | 98 | 42 | 127 | 8 |
| qwen3-8bstudent-8btioteacher | 13.10 | 0.625 | 0.539 | 0.500 | 90 | 41 | 189 | 81 |
| qwq-32b | 6.31 | 1.000 | 2.867 |  | 37 | 64 | 0 | 0 |
| tio-deepseek-r1-distill-llama-70b | 18.00 | 0.625 | 0.417 | 0.323 | 133 | 46 | 204 | 65 |
| tio-llama-3.3-70b-instruct | 10.81 | 1.000 | 1.668 |  | 98 | 75 | 0 | 0 |
| tio-qwen3-14b | 6.33 | 0.750 | 1.358 | 0.133 | 32 | 44 | 159 | 21 |
| tio-qwen3-14b-nothink | 5.67 | 0.188 | 1.444 | 0.576 | 7 | 10 | 388 | 197 |
| tio-qwen3-32b | 4.69 | 0.812 | 3.528 | 0.875 | 14 | 47 | 72 | 63 |
| tio-qwen3-32b-nothink | 7.12 | 1.000 | 1.456 |  | 49 | 65 | 0 | 0 |
| tio-qwen3-8b | 4.18 | 0.688 | 0.738 | 0.943 | 25 | 21 | 116 | 109 |
| tio-qwen3-8b-nothink | 6.56 | 0.562 | 0.820 | 0.596 | 35 | 24 | 204 | 111 |
| tio-qwq-32b | 5.79 | 0.875 | 4.000 | 1.062 | 11 | 70 | 44 | 46 |

## obj_4 / conjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 10.42 | 0.750 | 0.612 | 0.361 | 87 | 38 | 133 | 47 |
| llama-3.3-70b-instruct | 4.88 | 1.000 | 3.019 |  | 12 | 66 | 0 | 0 |
| qwen3-14b | 7.38 | 0.500 | 2.267 | 0.114 | 19 | 40 | 324 | 36 |
| qwen3-14b-nothink | 12.00 | 0.312 | 1.427 | 0.758 | 28 | 32 | 350 | 145 |
| qwen3-32b | 12.12 | 1.000 | 1.939 |  | 79 | 115 | 0 | 0 |
| qwen3-32b-nothink | 8.00 | 1.000 | 4.286 |  | 12 | 116 | 0 | 0 |
| qwen3-32bstudent-32btioteacher | 9.94 | 1.000 | 2.738 |  | 43 | 116 | 0 | 0 |
| qwen3-8b | 9.67 | 0.938 | 0.381 | 1.812 | 97 | 48 | 16 | 29 |
| qwen3-8b-nothink | 0.00 | 0.062 |  | 0.442 | 0 | 0 | 517 | 158 |
| qwen3-8bstudent-32bbaselineteacher | 18.75 | 0.750 | 0.557 | 0.212 | 149 | 76 | 150 | 30 |
| qwen3-8bstudent-32btioteacher | 6.17 | 0.375 | 1.667 | 0.084 | 18 | 19 | 415 | 34 |
| qwen3-8bstudent-8btioteacher | 14.00 | 0.500 | 0.282 | 0.056 | 92 | 20 | 341 | 19 |
| qwq-32b | 6.62 | 1.000 | 0.794 |  | 59 | 47 | 0 | 0 |
| tio-deepseek-r1-distill-llama-70b | 14.69 | 0.812 | 0.967 | 0.805 | 103 | 86 | 78 | 57 |
| tio-llama-3.3-70b-instruct | 5.94 | 1.000 | 0.700 |  | 19 | 76 | 0 | 0 |
| tio-qwen3-14b | 8.43 | 0.875 | 0.942 | 0.125 | 75 | 43 | 80 | 10 |
| tio-qwen3-14b-nothink | 12.62 | 0.500 | 1.456 | 1.045 | 43 | 58 | 196 | 164 |
| tio-qwen3-32b | 8.25 | 1.000 | 6.044 |  | 21 | 111 | 0 | 0 |
| tio-qwen3-32b-nothink | 10.93 | 0.938 | 7.226 | 3.500 | 21 | 143 | 10 | 35 |
| tio-qwen3-8b | 8.47 | 0.938 | 0.528 | 0.364 | 77 | 50 | 33 | 12 |
| tio-qwen3-8b-nothink | 18.80 | 0.312 | 0.044 | 0.137 | 90 | 4 | 438 | 57 |
| tio-qwq-32b | 5.38 | 1.000 | 1.278 |  | 30 | 56 | 0 | 0 |

## obj_4 / xor

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 7.75 | 0.250 | 0.363 | 0.483 | 23 | 8 | 371 | 168 |
| llama-3.3-70b-instruct | 13.00 | 0.750 | 0.664 | 0.393 | 97 | 59 | 130 | 50 |
| qwen3-14b | 8.58 | 0.750 | 2.048 | 0.281 | 49 | 54 | 141 | 39 |
| qwen3-14b-nothink | 23.00 | 0.188 | 1.230 | 0.905 | 31 | 38 | 330 | 255 |
| qwen3-32b | 13.67 | 0.562 | 1.142 | 0.188 | 68 | 55 | 267 | 48 |
| qwen3-32b-nothink | 16.40 | 0.938 | 1.913 | 0.250 | 149 | 97 | 36 | 9 |
| qwen3-32bstudent-32btioteacher | 7.69 | 1.000 | 2.179 |  | 52 | 71 | 0 | 0 |
| qwen3-8b | 9.00 | 0.250 | 0.237 | 0.545 | 29 | 7 | 363 | 177 |
| qwen3-8b-nothink |  | 0.000 |  | 1.523 | 0 | 0 | 396 | 324 |
| qwen3-8bstudent-32bbaselineteacher | 5.00 | 0.375 | 0.694 | 1.832 | 18 | 12 | 177 | 273 |
| qwen3-8bstudent-32btioteacher | 10.25 | 0.500 | 3.108 | 1.682 | 34 | 48 | 185 | 175 |
| qwen3-8bstudent-8btioteacher | 17.60 | 0.312 | 2.008 | 1.171 | 29 | 59 | 275 | 220 |
| qwq-32b | 12.19 | 1.000 | 0.912 |  | 105 | 90 | 0 | 0 |
| tio-deepseek-r1-distill-llama-70b | 21.00 | 0.250 | 0.805 | 0.410 | 54 | 28 | 392 | 148 |
| tio-llama-3.3-70b-instruct | 15.08 | 0.812 | 0.843 | 0.667 | 114 | 82 | 81 | 54 |
| tio-qwen3-14b | 8.18 | 0.688 | 1.491 | 0.483 | 40 | 50 | 160 | 65 |
| tio-qwen3-14b-nothink |  | 0.000 |  | 0.802 | 0 | 0 | 447 | 273 |
| tio-qwen3-32b | 8.93 | 0.938 | 1.867 | 0.667 | 56 | 78 | 27 | 18 |
| tio-qwen3-32b-nothink | 19.83 | 0.750 | 1.610 | 0.401 | 125 | 113 | 138 | 42 |
| tio-qwen3-8b | 5.67 | 0.188 | 2.143 | 1.650 | 8 | 9 | 223 | 362 |
| tio-qwen3-8b-nothink |  | 0.000 |  | 0.276 | 0 | 0 | 576 | 144 |
| tio-qwq-32b | 13.62 | 0.812 | 2.884 | 0.351 | 65 | 112 | 100 | 35 |

## obj_8 / disjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 13.33 | 0.188 | 0.602 | 0.386 | 27 | 11 | 457 | 128 |
| llama-3.3-70b-instruct | 19.07 | 0.875 | 1.512 | 0.185 | 117 | 149 | 76 | 14 |
| qwen3-14b | 7.43 | 0.438 | 2.494 | 0.145 | 18 | 34 | 357 | 48 |
| qwen3-14b-nothink | 11.25 | 0.250 | 1.530 | 4.474 | 19 | 26 | 330 | 210 |
| qwen3-32b | 7.60 | 0.625 | 1.007 | 1.057 | 23 | 53 | 133 | 137 |
| qwen3-32b-nothink | 19.18 | 0.688 | 0.784 | 0.620 | 111 | 100 | 141 | 84 |
| qwen3-32bstudent-32btioteacher | 9.14 | 0.875 | 2.331 | 0.169 | 47 | 81 | 77 | 13 |
| qwen3-8b |  | 0.000 |  | 2.601 | 0 | 0 | 223 | 497 |
| qwen3-8b-nothink | 14.00 | 0.188 | 2.324 | 0.124 | 21 | 21 | 525 | 60 |
| qwen3-8bstudent-32bbaselineteacher | 22.33 | 0.375 | 0.926 | 1.174 | 66 | 68 | 305 | 145 |
| qwen3-8bstudent-32btioteacher | 13.67 | 0.188 | 1.529 | 0.400 | 23 | 18 | 393 | 192 |
| qwen3-8bstudent-8btioteacher | 25.00 | 0.125 | 0.303 | 0.201 | 41 | 9 | 518 | 81 |
| qwq-32b | 13.00 | 0.938 | 1.748 | 0.216 | 85 | 110 | 37 | 8 |
| tio-deepseek-r1-distill-llama-70b | 29.67 | 0.188 | 0.536 | 0.303 | 62 | 27 | 454 | 131 |
| tio-llama-3.3-70b-instruct | 18.79 | 0.875 | 2.394 | 0.231 | 97 | 166 | 74 | 16 |
| tio-qwen3-14b | 9.33 | 0.562 | 2.095 | 0.191 | 30 | 54 | 265 | 50 |
| tio-qwen3-14b-nothink | 18.00 | 0.062 | 1.250 | 0.921 | 8 | 10 | 390 | 285 |
| tio-qwen3-32b | 12.80 | 0.938 | 2.241 | 0.957 | 75 | 117 | 23 | 22 |
| tio-qwen3-32b-nothink | 16.71 | 0.438 | 1.058 | 0.535 | 35 | 82 | 279 | 126 |
| tio-qwen3-8b | 19.50 | 0.125 | 1.080 | 0.993 | 19 | 20 | 330 | 300 |
| tio-qwen3-8b-nothink | 20.14 | 0.438 | 0.814 | 0.413 | 104 | 37 | 336 | 69 |
| tio-qwq-32b | 25.00 | 0.875 | 4.412 | 1.400 | 123 | 227 | 40 | 50 |

## obj_8 / conjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 24.00 | 0.188 | 0.503 | 0.721 | 49 | 23 | 370 | 206 |
| llama-3.3-70b-instruct | 15.50 | 0.750 | 1.456 | 1.390 | 88 | 98 | 111 | 69 |
| qwen3-14b | 36.67 | 0.188 | 0.873 | 0.524 | 59 | 51 | 421 | 164 |
| qwen3-14b-nothink | 5.00 | 0.062 |  | 1.098 | 0 | 5 | 460 | 215 |
| qwen3-32b | 17.80 | 0.312 | 1.836 | 1.511 | 33 | 56 | 230 | 265 |
| qwen3-32b-nothink | 17.00 | 0.438 | 6.151 | 0.817 | 30 | 89 | 246 | 159 |
| qwen3-32bstudent-32btioteacher | 19.62 | 0.500 | 3.930 | 1.317 | 46 | 111 | 171 | 189 |
| qwen3-8b | 3.67 | 0.188 | 0.750 | 1.608 | 7 | 4 | 293 | 292 |
| qwen3-8b-nothink | 7.00 | 0.062 | 0.000 | 0.162 | 7 | 0 | 599 | 76 |
| qwen3-8bstudent-32bbaselineteacher | 22.40 | 0.312 | 1.446 | 1.739 | 62 | 50 | 320 | 175 |
| qwen3-8bstudent-32btioteacher | 20.50 | 0.125 | 2.875 | 0.713 | 12 | 29 | 360 | 270 |
| qwen3-8bstudent-8btioteacher | 14.20 | 0.625 | 1.839 | 1.728 | 55 | 87 | 113 | 157 |
| qwq-32b | 20.55 | 0.688 | 0.847 | 0.705 | 115 | 111 | 134 | 91 |
| tio-deepseek-r1-distill-llama-70b | 7.00 | 0.125 | 0.375 | 0.938 | 10 | 4 | 342 | 288 |
| tio-llama-3.3-70b-instruct | 14.31 | 1.000 | 1.394 |  | 105 | 124 | 0 | 0 |
| tio-qwen3-14b | 34.22 | 0.562 | 1.145 | 0.232 | 172 | 136 | 258 | 57 |
| tio-qwen3-14b-nothink | 5.00 | 0.062 |  | 1.212 | 0 | 5 | 352 | 323 |
| tio-qwen3-32b | 19.90 | 0.625 | 2.591 | 1.640 | 71 | 128 | 114 | 156 |
| tio-qwen3-32b-nothink | 22.29 | 0.438 | 1.647 | 1.865 | 61 | 95 | 167 | 238 |
| tio-qwen3-8b | 4.50 | 0.125 | 0.917 | 0.995 | 5 | 4 | 367 | 263 |
| tio-qwen3-8b-nothink | 8.00 | 0.062 | 0.143 | 0.791 | 7 | 1 | 577 | 98 |
| tio-qwq-32b | 19.79 | 0.875 | 2.259 | 7.750 | 128 | 149 | 21 | 69 |

## obj_8 / xor

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 20.38 | 0.500 | 0.513 | 0.696 | 111 | 50 | 256 | 104 |
| llama-3.3-70b-instruct | 17.82 | 0.688 | 0.953 | 0.580 | 111 | 85 | 158 | 67 |
| qwen3-14b | 11.67 | 0.562 | 2.311 | 0.249 | 46 | 59 | 253 | 62 |
| qwen3-14b-nothink | 22.00 | 0.125 | 0.696 | 2.675 | 26 | 18 | 306 | 324 |
| qwen3-32b | 22.60 | 0.625 | 0.693 | 0.571 | 142 | 84 | 183 | 87 |
| qwen3-32b-nothink | 26.11 | 0.562 | 0.884 | 1.339 | 130 | 105 | 181 | 134 |
| qwen3-32bstudent-32btioteacher | 13.07 | 0.938 | 1.357 | 1.045 | 94 | 102 | 22 | 23 |
| qwen3-8b | 10.75 | 0.250 | 0.525 | 0.648 | 31 | 12 | 348 | 192 |
| qwen3-8b-nothink |  | 0.000 |  | 0.360 | 0 | 0 | 564 | 156 |
| qwen3-8bstudent-32bbaselineteacher | 16.67 | 0.188 | 1.458 | 1.847 | 22 | 28 | 271 | 314 |
| qwen3-8bstudent-32btioteacher | 17.50 | 0.250 | 0.865 | 1.228 | 41 | 29 | 264 | 276 |
| qwen3-8bstudent-8btioteacher | 14.40 | 0.312 | 0.549 | 0.739 | 46 | 26 | 297 | 198 |
| qwq-32b | 11.50 | 0.750 | 3.371 | 0.533 | 56 | 82 | 120 | 60 |
| tio-deepseek-r1-distill-llama-70b | 31.33 | 0.188 | 0.334 | 0.453 | 71 | 23 | 427 | 158 |
| tio-llama-3.3-70b-instruct | 16.36 | 0.875 | 1.423 | 0.651 | 110 | 119 | 54 | 35 |
| tio-qwen3-14b | 14.93 | 0.875 | 1.844 | 10.976 | 99 | 110 | 33 | 57 |
| tio-qwen3-14b-nothink | 23.00 | 0.062 | 0.769 | 0.970 | 13 | 10 | 382 | 293 |
| tio-qwen3-32b | 16.73 | 0.938 | 2.321 | 1.368 | 102 | 149 | 19 | 26 |
| tio-qwen3-32b-nothink | 21.88 | 0.500 | 5.558 | 4.045 | 67 | 108 | 166 | 194 |
| tio-qwen3-8b | 18.57 | 0.438 | 0.614 | 0.545 | 84 | 46 | 272 | 133 |
| tio-qwen3-8b-nothink |  | 0.000 |  | 0.193 | 0 | 0 | 610 | 110 |
| tio-qwq-32b | 21.20 | 0.625 | 1.830 | 4.592 | 78 | 134 | 106 | 164 |
