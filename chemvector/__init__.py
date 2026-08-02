"""ChemVector image-to-chemical-structure model.

The package is physically organized by lifecycle and domain. Additional search
paths preserve historical imports such as ``chemvector.model`` and module CLI
entries such as ``python -m chemvector.train``.
"""

from pathlib import Path
from typing import TYPE_CHECKING


_PACKAGE_ROOT = Path(__file__).resolve().parent
_COMPATIBILITY_SEARCH_ROOTS = (
    "core",
    "training",
    "evaluation",
    "inference",
    "tools",
    "workflows",
    "components/atoms",
    "components/bonds",
    "components/evidence",
    "components/graph",
    "components/ocr",
    "components/vision",
)


def _append_search_path(path: Path) -> None:
    value = str(path)
    if path.is_dir() and value not in __path__:
        __path__.append(value)


for _relative_path in _COMPATIBILITY_SEARCH_ROOTS:
    _root = _PACKAGE_ROOT / _relative_path
    _append_search_path(_root)
    if _relative_path in {"training", "evaluation", "tools"} or _relative_path.startswith(
        "components/"
    ):
        for _child in sorted(_root.rglob("*")):
            if _child.is_dir() and "__pycache__" not in _child.parts:
                _append_search_path(_child)

if TYPE_CHECKING:
    from chemvector.model import ChemVectorConfig, ChemVectorModel

__all__ = ["ChemVectorConfig", "ChemVectorModel"]


def __getattr__(name: str):
    if name in __all__:
        from chemvector.model import ChemVectorConfig, ChemVectorModel

        return {
            "ChemVectorConfig": ChemVectorConfig,
            "ChemVectorModel": ChemVectorModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
