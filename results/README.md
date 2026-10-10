# Local Results

Generated traces and checkpoints are excluded from Git. Compact evidence used
for writing is tracked under `evidence/` and `docs/phase2_writing_bundle/`.

Retained Phase 3 roots are the completed Qwen, Llama, Gemma, and Phi development
runs under `phase3_arfa/nrp/`. Failed calibration attempts and superseded local
debug runs are not retained. The Gemma residual trace from the initial threshold
is stored as `traces.jsonl.gz` because one terminal observation made the
uncompressed JSONL unusually large; the selected threshold trace remains in its
calibration directory. Compression is lossless.
