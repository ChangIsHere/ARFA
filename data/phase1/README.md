# Phase 1 Data Workspace

This directory is a local workspace for Phase 1 data.

Expected local-only contents:

```text
external_intercode/      official InterCode repository clone, ignored by git
final/*.jsonl            ARFA Phase 1 records derived from external trajectories, ignored by git
annotation/*.json        annotation/self-agreement outputs, ignored by git
```

External source:

```text
https://github.com/princeton-nlp/intercode
```

Do not commit downloaded external datasets or generated JSONL data files unless the repository policy changes.
