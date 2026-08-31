# Summary judge report

Models found: 22

## obj_4 / disjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 12.85 | 0.812 | 1.046 | 0.487 | 132 | 35 | 105 | 30 |
| llama-3.3-70b-instruct | 7.75 | 1.000 | 2.390 |  | 46 | 78 | 0 | 0 |
| qwen3-14b | 5.73 | 0.688 | 2.221 | 0.072 | 27 | 36 | 210 | 15 |
| qwen3-14b-nothink | 7.60 | 0.625 | 1.485 | 0.444 | 37 | 39 | 194 | 76 |
| qwen3-32b | 9.73 | 0.688 | 1.758 | 0.261 | 76 | 31 | 185 | 40 |
| qwen3-32b-nothink | 12.82 | 0.688 | 1.525 | 0.254 | 87 | 54 | 180 | 45 |
| qwen3-32bstudent-32btioteacher | 5.13 | 0.938 | 1.313 | 0.216 | 27 | 50 | 37 | 8 |
| qwen3-8b | 5.73 | 0.688 | 0.600 | 0.093 | 48 | 15 | 206 | 19 |
| qwen3-8b-nothink | 6.14 | 0.438 | 2.816 | 1.825 | 13 | 30 | 163 | 242 |
| qwen3-8bstudent-32bbaselineteacher | 11.25 | 0.250 | 1.117 | 0.139 | 30 | 15 | 476 | 63 |
| qwen3-8bstudent-32btioteacher | 12.44 | 0.562 | 0.772 | 0.091 | 85 | 27 | 289 | 26 |
| qwen3-8bstudent-8btioteacher | 9.50 | 0.750 | 0.468 | 0.108 | 88 | 26 | 164 | 16 |
| qwq-32b | 7.50 | 0.625 | 1.167 | 0.405 | 30 | 45 | 194 | 76 |
| tio-deepseek-r1-distill-llama-70b | 11.62 | 0.812 | 1.953 | 0.385 | 81 | 70 | 99 | 36 |
| tio-llama-3.3-70b-instruct | 8.38 | 1.000 | 1.583 |  | 61 | 73 | 0 | 0 |
| tio-qwen3-14b | 4.62 | 0.500 | 1.238 | 0.175 | 17 | 20 | 307 | 53 |
| tio-qwen3-14b-nothink | 6.00 | 0.438 | 1.627 | 0.725 | 21 | 21 | 263 | 142 |
| tio-qwen3-32b | 5.62 | 0.812 | 2.371 | 0.135 | 20 | 53 | 119 | 16 |
| tio-qwen3-32b-nothink | 9.44 | 0.562 | 1.690 | 2.135 | 46 | 39 | 107 | 208 |
| tio-qwen3-8b | 9.00 | 0.875 | 0.429 | 0.169 | 97 | 29 | 77 | 13 |
| tio-qwen3-8b-nothink | 4.22 | 0.562 | 2.444 | 0.678 | 12 | 26 | 199 | 116 |
| tio-qwq-32b | 9.21 | 0.875 | 3.200 | 0.552 | 49 | 80 | 58 | 32 |

## obj_4 / conjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 8.78 | 0.562 | 0.372 | 0.488 | 55 | 23 | 212 | 103 |
| llama-3.3-70b-instruct | 7.44 | 1.000 | 2.602 |  | 25 | 94 | 0 | 0 |
| qwen3-14b | 5.60 | 0.312 | 1.917 | 0.068 | 12 | 16 | 465 | 30 |
| qwen3-14b-nothink | 7.71 | 0.438 | 2.429 | 0.566 | 22 | 32 | 317 | 88 |
| qwen3-32b | 7.45 | 0.688 | 1.380 | 0.299 | 54 | 28 | 174 | 51 |
| qwen3-32b-nothink | 13.33 | 0.938 | 1.567 | 4.000 | 24 | 176 | 9 | 36 |
| qwen3-32bstudent-32btioteacher | 9.00 | 0.875 | 0.679 | 0.364 | 25 | 101 | 66 | 24 |
| qwen3-8b | 10.43 | 0.438 | 0.636 | 0.148 | 48 | 25 | 361 | 44 |
| qwen3-8b-nothink | 8.00 | 0.062 | 0.600 | 0.150 | 5 | 3 | 596 | 79 |
| qwen3-8bstudent-32bbaselineteacher | 12.11 | 0.562 | 0.860 | 0.129 | 75 | 34 | 279 | 36 |
| qwen3-8bstudent-32btioteacher | 8.11 | 0.562 | 1.433 | 0.287 | 36 | 37 | 245 | 70 |
| qwen3-8bstudent-8btioteacher | 7.75 | 0.500 | 0.952 | 0.129 | 32 | 30 | 320 | 40 |
| qwq-32b | 9.62 | 1.000 | 1.302 |  | 83 | 71 | 0 | 0 |
| tio-deepseek-r1-distill-llama-70b | 14.08 | 0.750 | 1.091 | 0.877 | 79 | 90 | 96 | 84 |
| tio-llama-3.3-70b-instruct | 8.12 | 1.000 | 4.348 |  | 36 | 94 | 0 | 0 |
| tio-qwen3-14b | 8.29 | 0.438 | 1.017 | 0.115 | 29 | 29 | 364 | 41 |
| tio-qwen3-14b-nothink | 8.80 | 0.625 | 2.056 | 2.005 | 25 | 63 | 94 | 176 |
| tio-qwen3-32b | 8.87 | 0.938 | 1.990 | 0.500 | 20 | 113 | 30 | 15 |
| tio-qwen3-32b-nothink | 8.80 | 0.625 |  | 2.213 | 0 | 88 | 92 | 178 |
| tio-qwen3-8b | 9.80 | 0.938 | 0.812 | 0.957 | 83 | 64 | 23 | 22 |
| tio-qwen3-8b-nothink |  | 0.000 |  | 0.284 | 0 | 0 | 578 | 142 |
| tio-qwq-32b | 7.06 | 1.000 | 1.021 |  | 57 | 56 | 0 | 0 |

## obj_4 / xor

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 5.33 | 0.375 | 0.778 | 0.226 | 18 | 14 | 370 | 79 |
| llama-3.3-70b-instruct |  | 0.000 |  | 0.757 | 0 | 0 | 428 | 292 |
| qwen3-14b | 7.44 | 0.562 | 1.687 | 0.323 | 32 | 35 | 242 | 73 |
| qwen3-14b-nothink | 4.33 | 0.375 | 0.917 | 0.709 | 14 | 12 | 313 | 137 |
| qwen3-32b | 9.29 | 0.438 | 0.667 | 0.649 | 39 | 26 | 252 | 153 |
| qwen3-32b-nothink | 4.80 | 0.312 | 3.800 | 0.852 | 5 | 19 | 291 | 204 |
| qwen3-32bstudent-32btioteacher | 6.17 | 0.750 | 2.056 | 0.530 | 30 | 44 | 123 | 57 |
| qwen3-8b | 20.00 | 0.125 | 1.892 | 1.545 | 19 | 21 | 384 | 246 |
| qwen3-8b-nothink | 14.00 | 0.062 | 1.333 | 0.622 | 6 | 8 | 452 | 223 |
| qwen3-8bstudent-32bbaselineteacher | 9.40 | 0.312 | 0.554 | 1.275 | 29 | 18 | 261 | 234 |
| qwen3-8bstudent-32btioteacher | 12.70 | 0.625 | 0.713 | 1.304 | 69 | 58 | 163 | 107 |
| qwen3-8bstudent-8btioteacher | 5.00 | 0.125 | 0.250 | 0.911 | 8 | 2 | 348 | 282 |
| qwq-32b | 16.08 | 0.750 | 1.611 | 0.688 | 103 | 90 | 108 | 72 |
| tio-deepseek-r1-distill-llama-70b | 22.00 | 0.500 | 4.306 | 0.574 | 42 | 134 | 241 | 119 |
| tio-llama-3.3-70b-instruct | 13.33 | 0.188 | 2.111 | 1.276 | 22 | 18 | 289 | 296 |
| tio-qwen3-14b | 5.67 | 0.375 | 2.083 | 0.719 | 13 | 21 | 277 | 173 |
| tio-qwen3-14b-nothink | 19.00 | 0.250 | 1.375 | 0.951 | 32 | 44 | 314 | 226 |
| tio-qwen3-32b | 10.38 | 0.812 | 2.055 | 1.111 | 54 | 81 | 65 | 70 |
| tio-qwen3-32b-nothink | 11.38 | 0.500 | 1.615 | 1.027 | 36 | 55 | 181 | 179 |
| tio-qwen3-8b | 19.33 | 0.375 | 0.371 | 1.124 | 74 | 42 | 254 | 196 |
| tio-qwen3-8b-nothink | 13.00 | 0.062 | 1.167 | 0.330 | 6 | 7 | 529 | 146 |
| tio-qwq-32b | 19.08 | 0.750 | 1.263 | 0.990 | 104 | 125 | 92 | 88 |

## obj_8 / disjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 21.20 | 0.312 | 0.467 | 0.223 | 74 | 32 | 407 | 88 |
| llama-3.3-70b-instruct | 16.23 | 0.812 | 1.403 | 0.572 | 98 | 112 | 88 | 47 |
| qwen3-14b | 12.00 | 0.625 | 2.045 | 0.114 | 48 | 72 | 243 | 27 |
| qwen3-14b-nothink | 14.00 | 0.250 | 1.242 | 0.987 | 35 | 21 | 344 | 196 |
| qwen3-32b | 20.40 | 0.625 | 0.963 | 0.564 | 105 | 99 | 179 | 91 |
| qwen3-32b-nothink | 12.67 | 0.375 | 2.660 | 1.364 | 35 | 41 | 258 | 192 |
| qwen3-32bstudent-32btioteacher | 13.25 | 0.750 | 2.897 | 0.419 | 74 | 85 | 128 | 52 |
| qwen3-8b | 26.00 | 0.125 | 0.694 | 0.183 | 30 | 22 | 539 | 91 |
| qwen3-8b-nothink | 11.00 | 0.188 | 1.133 | 0.174 | 24 | 9 | 503 | 82 |
| qwen3-8bstudent-32bbaselineteacher | 13.60 | 0.312 | 1.079 | 0.347 | 38 | 30 | 376 | 118 |
| qwen3-8bstudent-32btioteacher | 30.00 | 0.125 | 0.557 | 0.544 | 39 | 21 | 419 | 209 |
| qwen3-8bstudent-8btioteacher | 9.00 | 0.062 | 1.250 | 0.458 | 4 | 5 | 513 | 162 |
| qwq-32b | 11.14 | 0.875 | 2.691 | 1.596 | 56 | 100 | 39 | 51 |
| tio-deepseek-r1-distill-llama-70b | 14.75 | 0.250 | 1.828 | 0.314 | 39 | 20 | 419 | 121 |
| tio-llama-3.3-70b-instruct | 17.64 | 0.688 | 2.197 | 0.555 | 70 | 124 | 151 | 74 |
| tio-qwen3-14b | 9.82 | 0.688 | 2.654 | 0.848 | 34 | 74 | 162 | 63 |
| tio-qwen3-14b-nothink | 12.67 | 0.188 | 2.500 | 0.829 | 11 | 27 | 353 | 232 |
| tio-qwen3-32b | 14.50 | 1.000 | 2.521 |  | 92 | 140 | 0 | 0 |
| tio-qwen3-32b-nothink | 19.43 | 0.438 | 1.593 | 0.635 | 57 | 79 | 256 | 149 |
| tio-qwen3-8b | 20.33 | 0.188 | 0.322 | 0.712 | 44 | 17 | 409 | 176 |
| tio-qwen3-8b-nothink | 19.20 | 0.312 | 1.061 | 0.487 | 63 | 33 | 353 | 142 |
| tio-qwq-32b | 13.56 | 0.562 | 2.728 | 1.721 | 58 | 64 | 135 | 180 |

## obj_8 / conjunctive

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 18.50 | 0.125 | 0.675 | 0.775 | 25 | 12 | 366 | 264 |
| llama-3.3-70b-instruct | 19.33 | 0.750 | 1.454 | 0.444 | 108 | 124 | 127 | 53 |
| qwen3-14b | 5.00 | 0.062 |  | 0.400 | 0 | 5 | 488 | 187 |
| qwen3-14b-nothink | 8.00 | 0.062 | 3.000 | 0.658 | 2 | 6 | 475 | 200 |
| qwen3-32b | 25.20 | 0.312 | 1.075 | 0.921 | 62 | 64 | 287 | 207 |
| qwen3-32b-nothink | 14.33 | 0.375 | 2.094 | 2.064 | 31 | 55 | 176 | 274 |
| qwen3-32bstudent-32btioteacher | 28.25 | 0.250 | 3.626 | 0.945 | 42 | 71 | 298 | 242 |
| qwen3-8b | 9.00 | 0.062 | 0.125 | 0.963 | 8 | 1 | 423 | 252 |
| qwen3-8b-nothink |  | 0.000 |  | 0.686 | 0 | 0 | 561 | 159 |
| qwen3-8bstudent-32bbaselineteacher | 10.75 | 0.250 | 2.188 | 0.591 | 21 | 22 | 351 | 189 |
| qwen3-8bstudent-32btioteacher | 3.00 | 0.062 | 2.000 | 0.898 | 1 | 2 | 408 | 267 |
| qwen3-8bstudent-8btioteacher | 21.50 | 0.125 | 4.625 | 1.806 | 6 | 37 | 400 | 230 |
| qwq-32b | 21.88 | 0.500 | 1.938 | 1.329 | 82 | 93 | 188 | 172 |
| tio-deepseek-r1-distill-llama-70b | 22.00 | 0.312 | 0.798 | 1.133 | 63 | 47 | 257 | 238 |
| tio-llama-3.3-70b-instruct | 19.82 | 0.688 | 2.064 | 0.718 | 105 | 113 | 138 | 87 |
| tio-qwen3-14b | 22.86 | 0.438 | 1.534 | 0.384 | 59 | 101 | 300 | 105 |
| tio-qwen3-14b-nothink |  | 0.000 |  | 0.747 | 0 | 0 | 436 | 284 |
| tio-qwen3-32b | 26.75 | 0.500 | 2.494 | 1.441 | 65 | 149 | 160 | 200 |
| tio-qwen3-32b-nothink | 15.25 | 0.250 | 3.444 | 3.577 | 17 | 44 | 176 | 364 |
| tio-qwen3-8b | 5.00 | 0.125 | 0.500 | 0.955 | 8 | 2 | 408 | 222 |
| tio-qwen3-8b-nothink |  | 0.000 |  | 0.465 | 0 | 0 | 603 | 117 |
| tio-qwq-32b | 20.92 | 0.812 | 1.461 | 1.440 | 128 | 144 | 61 | 74 |

## obj_8 / xor

| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-r1-distill-llama-70b | 15.00 | 0.250 | 0.892 | 0.273 | 37 | 23 | 427 | 113 |
| llama-3.3-70b-instruct | 18.00 | 0.312 | 1.004 | 0.487 | 52 | 38 | 337 | 158 |
| qwen3-14b | 16.83 | 0.375 | 3.677 | 0.470 | 44 | 57 | 322 | 128 |
| qwen3-14b-nothink | 4.00 | 0.062 | 1.000 | 1.116 | 2 | 2 | 336 | 339 |
| qwen3-32b | 15.33 | 0.188 | 2.292 | 0.478 | 18 | 28 | 412 | 173 |
| qwen3-32b-nothink | 18.80 | 0.312 | 2.592 | 0.995 | 34 | 60 | 274 | 221 |
| qwen3-32bstudent-32btioteacher | 23.00 | 0.188 | 0.944 | 0.487 | 47 | 22 | 398 | 187 |
| qwen3-8b | 25.00 | 0.062 | 0.667 | 0.949 | 15 | 10 | 380 | 294 |
| qwen3-8b-nothink |  | 0.000 |  | 0.681 | 0 | 0 | 582 | 138 |
| qwen3-8bstudent-32bbaselineteacher | 12.00 | 0.062 | 1.400 | 0.681 | 5 | 7 | 385 | 217 |
| qwen3-8bstudent-32btioteacher | 6.00 | 0.062 | 1.000 | 0.485 | 3 | 3 | 466 | 209 |
| qwen3-8bstudent-8btioteacher |  | 0.000 |  | 0.617 | 0 | 0 | 518 | 202 |
| qwq-32b | 20.25 | 0.500 | 3.039 | 0.936 | 71 | 91 | 209 | 151 |
| tio-deepseek-r1-distill-llama-70b | 15.00 | 0.125 | 1.543 | 0.504 | 12 | 18 | 442 | 188 |
| tio-llama-3.3-70b-instruct | 22.43 | 0.438 | 1.098 | 0.697 | 88 | 69 | 250 | 155 |
| tio-qwen3-14b | 17.89 | 0.562 | 2.366 | 0.551 | 74 | 87 | 210 | 105 |
| tio-qwen3-14b-nothink | 41.00 | 0.062 | 0.640 | 0.941 | 25 | 16 | 412 | 263 |
| tio-qwen3-32b | 20.82 | 0.688 | 1.258 | 0.825 | 108 | 121 | 125 | 100 |
| tio-qwen3-32b-nothink | 19.67 | 0.188 | 4.230 | 1.022 | 24 | 35 | 313 | 272 |
| tio-qwen3-8b |  | 0.000 |  | 1.381 | 0 | 0 | 359 | 361 |
| tio-qwen3-8b-nothink |  | 0.000 |  | 0.600 | 0 | 0 | 555 | 165 |
| tio-qwq-32b | 18.00 | 0.438 | 1.869 | 1.485 | 56 | 70 | 194 | 211 |
