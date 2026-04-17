import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "conway_life.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("conway_life", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _side_lengths(poly):
    lengths = []
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return lengths


def test_penrose_tiling_uses_only_rhombi():
    module = _load_module()
    polys, face_types, bbox = module.generate_penrose_tiling(4)

    assert len(polys) == 160
    assert len(face_types) == len(polys)
    assert bbox[0] > 0 and bbox[1] > 0
    assert set(face_types) == {0, 1}

    for poly in polys:
        assert len(poly) == 4
        lengths = _side_lengths(poly)
        assert max(lengths) - min(lengths) < 1e-6
