# Phase 1.5 Pilot Evidence

This bundle makes the engineering collection audit inspectable without publishing private traces or labels.

- `pilot_audit.json` records collection-only counts and gate results.
- `pilot_blind_packet_unlabeled.jsonl` contains 23 content-deduplicated, unlabeled commitments and observations.
- `pilot_source_index_redacted.jsonl` contains 25 source-step references, with task IDs hashed and model identity removed.
- `annotation_manifest.json` records packet construction counts and schema.
- `artifact_manifest.json` records SHA-256 hashes, byte sizes, and record counts for private source artifacts, protocol files, and published evidence.

This bundle contains no residual-effect result. The 28% provisional fast-candidate rate is a collector diagnostic, not a validated routing metric. Phase 3 remains locked.
