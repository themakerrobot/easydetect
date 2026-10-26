# Pretrained weights and the mirror

Back to the [README](../README.md).

## Pretrained weights

`Detector("ed-s")` downloads the IR for that name on first use and caches it
in `~/.easydetect/` (`$EASYDETECT_HOME` to move it, `$EASYDETECT_ASSETS_URL` to
point at an internal mirror — handy for air-gapped sites). Known names:
`ed-n`, `ed-s`, `ed-m`, `ed-l`, `ed-x`; each mirror entry is
`<name>/<name>.xml`, `.bin` (OpenVINO), `.onnx` (ONNX Runtime), `.pt` and
`labels.txt`, on
[huggingface.co/leeyunjai/easydetect](https://huggingface.co/leeyunjai/easydetect).

The weights are the original COCO checkpoints, which their authors release
under Apache-2.0 — only the COCO-trained ones; the Objects365-pretrained
checkpoints may carry that dataset's terms and are not used. This package's
network matches the reference module for module, so they load with
`strict=True` — no remapping and no re-training.

## Building and uploading the mirror

The **mirror** workflow in GitHub Actions does it all: it converts every size,
exports the IRs, checks that each one finds the people in a test picture,
uploads to Hugging Face, and downloads one back the way users will. It needs one
repository secret, `HF_TOKEN` (a Hugging Face token with write access); run it
from *Actions → mirror → Run workflow*, with the sizes to build, or tick
*readme_only* to refresh the repo's front page from
[`tools/hub_README.md`](../tools/hub_README.md).

By hand, the same thing takes a few CPU-minutes:

```bash
python tools/build_mirror.py --out mirror   # n s m l x: .pt + IR + labels + README

pip install -U "huggingface_hub[cli]"       # ships the `hf` command
hf auth login                               # a token with write access
hf upload leeyunjai/easydetect mirror . --repo-type=model
```

Fine-tuned models go under `models/<name>/`, each with its own card — the
[lab](https://github.com/themakerrobot/easydetect-lab)'s *Hugging Face* tab writes that folder for a finished run.

> Until the mirror has a size, `Detector("ed-x")` raises a
> `ModelNotFoundError` naming the ways forward — it never silently falls back to
> an untrained network. Point `$EASYDETECT_ASSETS_URL` at any host serving the
> same layout to use it now.

For a single checkpoint without the IR:

```bash
python tools/convert_upstream.py --size s --out ed-s.pt
```

