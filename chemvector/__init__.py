"""ChemVector image-to-chemical-structure model."""

from typing import TYPE_CHECKING

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
