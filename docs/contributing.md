# Development and releases

Back to the [README](../README.md).

## Development

```bash
pip install -e ".[train,dev]"
pytest -q          # the parity checklist lives in tests/
ruff check .
```

## Releasing

Bump the version in `pyproject.toml` and `easydetect/__init__.py` (and
`EASYDETECT_AT_LEAST` in [easydetect lab](https://github.com/themakerrobot/easydetect-lab)'s `run.py` when the lab needs it), add an
entry to `CHANGELOG.md`, then either push a `v*` tag or run the `publish`
workflow by hand with the version:

```bash
git tag v0.2.0
git push origin v0.2.0
```

`.github/workflows/publish.yml` uses **PyPI trusted publishing** — no API token
lives in the repo. One-time setup on PyPI (Publishing → add a pending publisher):

| Field | Value |
| --- | --- |
| PyPI project | `easydetect` |
| Owner | `themakerrobot` |
| Repository | `easydetect` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

The job refuses to publish when the tag and `pyproject.toml` version disagree, so
bump the version in the same commit you tag.

