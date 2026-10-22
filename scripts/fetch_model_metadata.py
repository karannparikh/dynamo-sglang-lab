#!/usr/bin/env python3
"""Fetch only the metadata a fake worker needs: config.json + tokenizer.

The fake engine loads no weights, but Dynamo's frontend still builds a real
ModelDeploymentCard -- it needs the tokenizer to detokenize and the config to
size things. Pointing ``--model-path`` at a local directory also makes
``dynamo.sglang`` skip its download entirely (``args.should_fetch_model``
returns False for a path that exists), so runs are offline and instant.

    python scripts/fetch_model_metadata.py Qwen/Qwen3-0.6B models/qwen3-0.6b [REVISION]

Downloads are pinned to a commit so every engineer gets byte-identical files;
``models/qwen3-0.6b`` in the repo was produced from ``QWEN3_0_6B_REVISION``.
"""

import sys
from pathlib import Path

# Everything a tokenizer + card needs, and nothing that carries weights.
WANTED = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "LICENSE",
)

QWEN3_0_6B_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_REVISIONS = {"Qwen/Qwen3-0.6B": QWEN3_0_6B_REVISION}


def main(repo_id: str, dest: str, revision: str | None = None) -> int:
    from huggingface_hub import hf_hub_download, list_repo_files

    revision = revision or DEFAULT_REVISIONS.get(repo_id)
    out = Path(dest).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    available = set(list_repo_files(repo_id, revision=revision))
    fetched = []
    for name in WANTED:
        if name not in available:
            continue
        path = hf_hub_download(repo_id=repo_id, filename=name, revision=revision)
        (out / name).write_bytes(Path(path).read_bytes())
        fetched.append(name)

    if "config.json" not in fetched:
        print(f"error: {repo_id} has no config.json", file=sys.stderr)
        return 1

    size_kb = sum((out / n).stat().st_size for n in fetched) / 1024
    print(f"wrote {len(fetched)} files ({size_kb:.0f} KiB) to {out}")
    print("  " + "\n  ".join(fetched))
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*sys.argv[1:]))
