# Qwen Routing Controls

Qwen2.5-Coder 7B/14B ran the same 100 development tasks under four routing
policies. This is supplementary evidence; it is separate from the native-pair
main table.

| Policy | Success | Tokens | Calls | Fast-call rate | Wall time/task |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual | 42% | 283,253 | 275 | 40.0% | 158.99s |
| Learned observation-only | 43% | 288,731 | 280 | 38.9% | 166.02s |
| Expectation gate-only | 38% | 279,066 | 274 | 46.0% | 154.67s |
| Budget-matched random | 41% | 285,784 | 278 | 41.0% | 169.17s |

Residual used fewer tokens than learned observation-only and random routing,
but their paired success intervals include zero. This run does not establish
that residual is superior to learned observation-only routing.
