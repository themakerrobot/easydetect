# Apache-2.0
"""The names easydetect lab imports from this package.

The lab lives in its own repository (themakerrobot/easydetect-lab) and runs on
the PyPI release, so renaming or moving one of these breaks it only after a
release. This test makes such a change visible here, before it ships.
"""

from __future__ import annotations

import importlib

import pytest

LAB_IMPORTS = {
    "easydetect": ["Detector", "__version__"],
    "easydetect.data.dataset": ["DetDataset", "list_images", "load_data_yaml"],
    "easydetect.data.labels": ["label_path", "label_row_to_box"],
    "easydetect.downloads": ["MODEL_NAMES", "assets_url"],
    "easydetect.plotting": ["draw_boxes"],
    "easydetect.results": ["Boxes"],
    "easydetect.sources": ["SourceLoader"],
    "easydetect.validator": ["validate_torch"],
}


@pytest.mark.parametrize("module", sorted(LAB_IMPORTS))
def test_the_lab_finds_what_it_imports(module):
    mod = importlib.import_module(module)
    missing = [name for name in LAB_IMPORTS[module] if not hasattr(mod, name)]
    assert not missing, f"easydetect lab imports {missing} from {module}"
