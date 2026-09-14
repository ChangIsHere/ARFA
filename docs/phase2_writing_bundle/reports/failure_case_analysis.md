# Phase 2 Failure-Case Analysis

Categories below are deterministic diagnostic labels derived from traces, not manual causal annotations.
Benchmark-environment anomalies are excluded in the gold-healthy sensitivity analysis.
Recovered JSON extraction is tracked separately and is not classified as a response-parse failure.

## Counts

| Model | Diagnostic category | Failures |
| --- | --- | ---: |
| arfa-qwen2.5-coder:7b-8k | benchmark environment anomaly | 5 |
| arfa-qwen2.5-coder:7b-8k | command execution error | 11 |
| arfa-qwen2.5-coder:7b-8k | incorrect command or semantics | 40 |
| arfa-qwen2.5-coder:7b-8k | max step exhaustion | 44 |
| arfa-qwen2.5-coder:7b-8k | premature termination | 39 |
| arfa-qwen2.5-coder:7b-8k | repeated action loop | 9 |
| arfa-llama3.1:8b-8k | benchmark environment anomaly | 5 |
| arfa-llama3.1:8b-8k | command execution error | 18 |
| arfa-llama3.1:8b-8k | incorrect command or semantics | 26 |
| arfa-llama3.1:8b-8k | max step exhaustion | 9 |
| arfa-llama3.1:8b-8k | premature termination | 50 |
| arfa-llama3.1:8b-8k | repeated action loop | 28 |
| arfa-llama3.1:8b-8k | response parse failure | 13 |
| arfa-qwen2.5-coder:14b-8k | benchmark environment anomaly | 5 |
| arfa-qwen2.5-coder:14b-8k | command execution error | 18 |
| arfa-qwen2.5-coder:14b-8k | incorrect command or semantics | 21 |
| arfa-qwen2.5-coder:14b-8k | max step exhaustion | 13 |
| arfa-qwen2.5-coder:14b-8k | premature termination | 61 |
| arfa-qwen2.5-coder:14b-8k | repeated action loop | 2 |

## Representative Cases

- `arfa-qwen2.5-coder:7b-8k` / `benchmark_environment_anomaly` / `intercode-nl2bash-fs1-012`: reward=0.340, calls=4; instruction: Compress regular files in the testbed directory tree that were last modified more than 7 days ago
- `arfa-qwen2.5-coder:7b-8k` / `command_execution_error` / `intercode-nl2bash-fs1-007`: reward=0.670, calls=3; instruction: Change directory to the directory containing the executable file of command "python"
- `arfa-qwen2.5-coder:7b-8k` / `incorrect_command_or_semantics` / `intercode-nl2bash-fs1-000`: reward=0.670, calls=3; instruction: Calculate a list of duplicate md5 sum hashes for all the ".java" files in the /testbed directory
- `arfa-qwen2.5-coder:7b-8k` / `max_step_exhaustion` / `intercode-nl2bash-fs1-006`: reward=0.700, calls=12; instruction: Calculate the total disk usage for each ".txt" file on the /testbed directory and prepend the system host name to the output
- `arfa-qwen2.5-coder:7b-8k` / `premature_termination` / `intercode-nl2bash-fs1-019`: reward=0.720, calls=2; instruction: Copy all files below the /testbed directory whose names contain "FooBar" to directory '/testbed/dir3/subdir1/subsubdir1/tmp'
- `arfa-qwen2.5-coder:7b-8k` / `repeated_action_loop` / `intercode-nl2bash-fs1-011`: reward=0.340, calls=5; instruction: Compress in parallel regular files in the testbed directory tree that were last modified more than 7 days ago
- `arfa-llama3.1:8b-8k` / `benchmark_environment_anomaly` / `intercode-nl2bash-fs1-012`: reward=0.340, calls=3; instruction: Compress regular files in the testbed directory tree that were last modified more than 7 days ago
- `arfa-llama3.1:8b-8k` / `command_execution_error` / `intercode-nl2bash-fs1-006`: reward=0.670, calls=8; instruction: Calculate the total disk usage for each ".txt" file on the /testbed directory and prepend the system host name to the output
- `arfa-llama3.1:8b-8k` / `incorrect_command_or_semantics` / `intercode-nl2bash-fs1-002`: reward=0.390, calls=3; instruction: Calculate the md5 sum of the contents of the sorted list of files "$FILES"
- `arfa-llama3.1:8b-8k` / `max_step_exhaustion` / `intercode-nl2bash-fs1-018`: reward=0.390, calls=12; instruction: Copies all files under the /testbed folder like "file.txt" with "FooBar" in the path to the root of the current folder, preserving mode, ownership and timestamp attributes.
- `arfa-llama3.1:8b-8k` / `premature_termination` / `intercode-nl2bash-fs1-000`: reward=0.670, calls=1; instruction: Calculate a list of duplicate md5 sum hashes for all the ".java" files in the /testbed directory
- `arfa-llama3.1:8b-8k` / `repeated_action_loop` / `intercode-nl2bash-fs1-019`: reward=0.390, calls=5; instruction: Copy all files below the /testbed directory whose names contain "FooBar" to directory '/testbed/dir3/subdir1/subsubdir1/tmp'
- `arfa-llama3.1:8b-8k` / `response_parse_failure` / `intercode-nl2bash-fs1-015`: reward=0.340, calls=6; instruction: Copies all files with "FooBar" in the path under the '/testbed/dir1' directory to the '/testbed/dir3/subdir1/subsubdir1/tmp' directory.
- `arfa-qwen2.5-coder:14b-8k` / `benchmark_environment_anomaly` / `intercode-nl2bash-fs1-012`: reward=0.470, calls=12; instruction: Compress regular files in the testbed directory tree that were last modified more than 7 days ago
- `arfa-qwen2.5-coder:14b-8k` / `command_execution_error` / `intercode-nl2bash-fs1-007`: reward=0.670, calls=5; instruction: Change directory to the directory containing the executable file of command "python"
- `arfa-qwen2.5-coder:14b-8k` / `incorrect_command_or_semantics` / `intercode-nl2bash-fs1-001`: reward=0.670, calls=4; instruction: Calculate md5 sum of the md5 sum of all the sorted files under /testbed/dir2/subdir2
- `arfa-qwen2.5-coder:14b-8k` / `max_step_exhaustion` / `intercode-nl2bash-fs1-034`: reward=0.820, calls=12; instruction: Count lines in each *.php file sorted by file in /testbed directory.
- `arfa-qwen2.5-coder:14b-8k` / `premature_termination` / `intercode-nl2bash-fs1-000`: reward=0.800, calls=2; instruction: Calculate a list of duplicate md5 sum hashes for all the ".java" files in the /testbed directory
- `arfa-qwen2.5-coder:14b-8k` / `repeated_action_loop` / `intercode-nl2bash-fs1-008`: reward=0.670, calls=5; instruction: Change permissions for all PHP files under the /testbed directory tree to 755 and print the number of files changed
