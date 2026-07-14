"""Byte-level text labels for the ChemVector OCR branch."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class OCRTextCodec:
    """Encode arbitrary UTF-8 labels without a chemistry-specific vocabulary."""

    max_length: int = 24

    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    BYTE_OFFSET = 3
    VOCAB_SIZE = BYTE_OFFSET + 256

    def __post_init__(self) -> None:
        if self.max_length < 3:
            raise ValueError("OCR max_length must allow BOS, one byte, and EOS")

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        payload = text.encode("utf-8")
        maximum_bytes = self.max_length - 2
        if len(payload) > maximum_bytes:
            raise ValueError(
                f"OCR label needs {len(payload)} bytes; maximum is {maximum_bytes}: {text!r}"
            )
        return [
            self.BOS_TOKEN_ID,
            *(byte + self.BYTE_OFFSET for byte in payload),
            self.EOS_TOKEN_ID,
        ]

    def decode(self, token_ids: list[int]) -> str:
        payload = bytearray()
        for token_id in token_ids:
            if token_id in {self.PAD_TOKEN_ID, self.BOS_TOKEN_ID}:
                continue
            if token_id == self.EOS_TOKEN_ID:
                break
            byte = token_id - self.BYTE_OFFSET
            if not 0 <= byte <= 255:
                raise ValueError(f"Invalid OCR byte token: {token_id}")
            payload.append(byte)
        return payload.decode("utf-8", errors="replace")


def _hydrogen_suffix(count: int) -> str:
    if count <= 0:
        return ""
    return "H" if count == 1 else f"H{count}"


def _charge_suffix(charge: int) -> str:
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    return sign if magnitude == 1 else f"{magnitude}{sign}"


def atom_display_text(atom: dict[str, Any]) -> str:
    """Approximate the atom label drawn by RDKit for the prepared dataset.

    Uncharged ordinary carbon is implicit in a skeletal formula. Other atoms use
    their element symbol followed by visible hydrogens, formal charge, and radical
    dots. The current main dataset excludes isotopes, but the prefix is retained so
    the same protocol can support later specialist data.
    """

    element = str(atom["element"])
    isotope = int(atom.get("isotope", 0))
    formal_charge = int(atom.get("formal_charge", 0))
    radical_electrons = int(atom.get("radical_electrons", 0))
    explicit_h = int(atom.get("explicit_h_count", 0))
    implicit_h = int(atom.get("implicit_h_count", 0))
    no_implicit = bool(atom.get("no_implicit", False))

    ordinary_carbon = (
        element == "C"
        and isotope == 0
        and formal_charge == 0
        and radical_electrons == 0
        and explicit_h == 0
        and not no_implicit
    )
    if ordinary_carbon:
        return ""

    hydrogen_count = explicit_h + implicit_h
    isotope_prefix = str(isotope) if isotope else ""
    radical_suffix = "." * radical_electrons
    return (
        isotope_prefix
        + element
        + _hydrogen_suffix(hydrogen_count)
        + _charge_suffix(formal_charge)
        + radical_suffix
    )


_ATOM_TEXT_PATTERN = re.compile(
    r"^(?P<isotope>\d+)?"
    r"(?P<element>Br|Cl|Se|Si|[BCFINOPS])"
    r"(?:H(?P<hydrogens>\d+)?)?"
    r"(?P<charge>(?:\d+)?[+-])?"
    r"(?P<radicals>\.*)$"
)


def parse_atom_display_text(text: str) -> dict[str, Any] | None:
    """Parse a standard visible atom label; return None for aliases or noise."""

    match = _ATOM_TEXT_PATTERN.fullmatch(text)
    if match is None:
        return None
    charge_text = match.group("charge")
    charge = 0
    if charge_text:
        sign = 1 if charge_text[-1] == "+" else -1
        magnitude = int(charge_text[:-1]) if charge_text[:-1] else 1
        charge = sign * magnitude
    hydrogen_text = match.group("hydrogens")
    has_hydrogen = "H" in text.removeprefix(match.group("isotope") or "")
    hydrogens = (int(hydrogen_text) if hydrogen_text else 1) if has_hydrogen else 0
    return {
        "element": match.group("element"),
        "formal_charge": charge,
        "isotope": int(match.group("isotope") or 0),
        "explicit_h_count": hydrogens,
        "radical_electrons": len(match.group("radicals")),
    }
