#!/usr/bin/env python3
# Apache-2.0
"""easydetect-hub's files named after their dataset, once: ``taco/ed-s-640/ed-s-640.onnx``
becomes ``taco/ed-s-640/taco-ed-s-640.onnx``, so models of several datasets can
share a folder once downloaded. hub_publish.py names new models this way; this
renames the ones published before.

    python tools/hub_dataset_names.py --dry-run     # what would change
    python tools/hub_dataset_names.py

Files are copied inside the repository (big ones are not downloaded) and the
old names removed in the same commit, one commit a model. Each model's
model.json and the hub's index.json follow, and the front README is rebuilt.
It holds hub_publish.py's lock the whole time, so a model published meanwhile
waits for it rather than writing the list over it. Running it again changes
nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hub_publish import REPO, TASKS, _fill_categories, _publishing, front_readme  # noqa: E402


def renamed(dataset: str, folder: str, name: str) -> str | None:
    """The file's new name, or None when it keeps its own (sample.jpg,
    model.json, a name already carrying the dataset)."""
    stem = f"{dataset}-{folder}"
    if name.startswith(stem):
        return None
    if name.startswith(folder) and name[len(folder):][:1] in (".", "-"):
        return stem + name[len(folder):]
    if any(head and name == head for _, head, _ in TASKS.values()):
        return f"{stem}.{name}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=REPO)
    p.add_argument("--stage", default="~/longrun/hub-stage",
                   help="hub_publish.py's --stage (its lock is taken there)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
        HfApi,
        hf_hub_download,
    )

    api = HfApi()
    with _publishing(Path(args.stage).expanduser()):
        index = json.loads(Path(hf_hub_download(args.repo, "index.json",
                                                force_download=True)).read_text())
        present = set(api.list_repo_files(args.repo))
        moved = 0
        for key, entry in sorted(index.items()):
            dataset, folder = key.split("/", 1)
            names = sorted(n[len(key) + 1:] for n in present if n.startswith(key + "/"))
            moves = {n: renamed(dataset, folder, n) for n in names}
            moves = {o: n for o, n in moves.items() if n}
            if not moves:
                continue
            ops = [CommitOperationCopy(src_path_in_repo=f"{key}/{o}", path_in_repo=f"{key}/{n}")
                   for o, n in moves.items()]
            ops += [CommitOperationDelete(path_in_repo=f"{key}/{o}") for o in moves]
            entry["files"] = [moves.get(f, f) for f in entry.get("files") or []]
            if f"{key}/model.json" in present:
                info = json.loads(Path(hf_hub_download(args.repo, f"{key}/model.json",
                                                       force_download=True)).read_text())
                info["files"] = [moves.get(f, f) for f in info.get("files") or []]
                ops.append(CommitOperationAdd(path_in_repo=f"{key}/model.json",
                                              path_or_fileobj=json.dumps(
                                                  info, ensure_ascii=False, indent=2).encode()))
            moved += 1
            print(f"{key}: " + ", ".join(f"{o} -> {n}" for o, n in moves.items()), flush=True)
            if not args.dry_run:
                api.create_commit(args.repo, operations=ops,
                                  commit_message=f"{key}: files named after {dataset}")
        if not moved:
            print("every file is named after its dataset already")
            return 0
        if args.dry_run:
            print(f"{moved} models would be renamed (dry run: nothing changed)")
            return 0
        _fill_categories(index)
        api.create_commit(args.repo, operations=[
            CommitOperationAdd(path_in_repo="index.json", path_or_fileobj=json.dumps(
                index, ensure_ascii=False, indent=2).encode()),
            CommitOperationAdd(path_in_repo="README.md",
                               path_or_fileobj=front_readme(index, args.repo).encode()),
        ], commit_message="index and front page: files named after their dataset")
    print(f"{moved} models renamed: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
