# Compatibility judge failures

Non-OK rows from `*_judge_compat.jsonl` (or missing compat files).

## llama-3.3-70b-instruct / obj_8 / disjunctive / trial 4

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 13 | 24 | 25 | ERROR | runtime_error: name 'generate_combinations' is not defined |

## llama-3.3-70b-instruct / obj_8 / xor / trial 13

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 31 | 60 | 61 | ERROR | runtime_error: cannot access local variable 'generate_combinations' where it is not associated with a value |
| 33 | 64 | 65 | ERROR | runtime_error: name 'list' is not defined |

## qwen3-32b / obj_8 / conjunctive / trial 9

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 38 | 74 | 75 | ERROR | runtime_error: name 'sorted' is not defined |

## qwen3-8b / obj_8 / xor / trial 4

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 28 | 54 | 55 | ERROR | runtime_error: 'int' object is not iterable |

## qwen3-8bstudent-32bbaselineteacher / obj_4 / disjunctive / trial 4

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 5 | 8 | 9 | ERROR | runtime_error: name 'zip' is not defined |

## qwen3-8bstudent-32bbaselineteacher / obj_8 / disjunctive / trial 13

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 26 | 50 | 51 | ERROR | runtime_error: name 'sorted' is not defined |

## qwen3-8bstudent-32btioteacher / obj_8 / disjunctive / trial 3

| index | announce_turn | test_turn | status | compile_error |
|---:|---:|---:|---|---|
| 15 | 28 | 29 | ERROR | runtime_error: name 'sorted' is not defined |
| 38 | 74 | 75 | ERROR | runtime_error: name 'sorted' is not defined |
