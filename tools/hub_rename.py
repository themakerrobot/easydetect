#!/usr/bin/env python3
# Apache-2.0
"""Rename the files on a Hugging Face repo: every path containing OLD gets NEW.

    python tools/hub_rename.py leeyunjai/easydetect dfine- ed- --readme tools/hub_README.md
    python tools/hub_rename.py leeyunjai/themaker .model/object/dfine- .model/object/ed-

Copies on the server (large files are not downloaded), then deletes the old
paths in a second commit, so the new ones exist before the old ones go.
"""

from __future__ import annotations

import argparse


def main() -> int:
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
        HfApi,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--readme", help="also upload this file as README.md")
    args = parser.parse_args()

    api = HfApi()
    moves = {f: f.replace(args.old, args.new) for f in api.list_repo_files(args.repo)
             if args.old in f}
    if not moves:
        print("nothing to rename")
        return 0
    for old, new in sorted(moves.items()):
        print(f"{old}  ->  {new}")
    ops = [CommitOperationCopy(src_path_in_repo=o, path_in_repo=n) for o, n in moves.items()]
    if args.readme:
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=args.readme))
    api.create_commit(args.repo, operations=ops, commit_message="Rename files")
    left = set(api.list_repo_files(args.repo))
    missing = [n for n in moves.values() if n not in left]
    if missing:
        raise SystemExit(f"not copied, old files kept: {missing}")
    api.create_commit(args.repo, operations=[CommitOperationDelete(path_in_repo=o) for o in moves],
                      commit_message="Remove the old names")
    print(f"{len(moves)} files renamed on {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
