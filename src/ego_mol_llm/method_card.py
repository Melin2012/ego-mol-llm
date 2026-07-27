"""Acquisition method card and RT / chemistry priors for annotation propagation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MethodCard:
    """
    Experimental context for the target/seed measurement.

    Prefer within-run relative RT over absolute library RT transfer.
    """

    chromatography: str = "RP-C18"
    polarity: str = "positive"  # positive | negative | both
    ionization: str = "ESI"
    gradient: str = "aqueous_to_organic"
    rt_unit: str = "seconds"
    instrument: str | None = None
    study_id: str | None = None
    notes: str | None = None
    # Soft RT windows for crude class priors on RP-C18 (seconds); optional
    early_rt_max: float = 180.0
    late_rt_min: float = 600.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MethodCard":
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def adduct_prior_ok(self, adduct: str | None) -> bool:
        if not adduct:
            return True
        a = adduct.replace(" ", "").lower()
        pol = (self.polarity or "both").lower()
        if pol.startswith("pos"):
            if a.endswith("-") and "m-h" in a.replace("+", ""):
                # allow [M-H]- only if polarity both; pure pos: penalize
                if "2m" not in a and "3m" not in a and a in {"[m-h]-", "m-h"}:
                    return False
            return True
        if pol.startswith("neg"):
            if a.endswith("+") and ("m+h" in a or "m+na" in a or "m+nh4" in a):
                return False
            return True
        return True


def rt_compatibility(
    seed_rt: float | None,
    neighbor_rts: list[float | None],
    *,
    card: MethodCard | None = None,
    name_hint: str | None = None,
) -> float:
    """
    Soft score in [0, 1] for RT / chromatography context.

    Uses relative RT to annotated neighbors when available; otherwise weak
    class priors on RP-C18 (polar early, lipids/late hydrophobics later).
    """
    card = card or MethodCard()
    if seed_rt is None:
        return 0.5  # neutral when unknown

    # Relative: prefer candidates whose peer neighbors are nearby in RT
    near = [r for r in neighbor_rts if r is not None]
    if near:
        med = sorted(near)[len(near) // 2]
        # typical peak width scale ~30–60 s; soft gaussian-ish
        d = abs(float(seed_rt) - float(med))
        if d <= 30:
            rel = 1.0
        elif d <= 90:
            rel = 0.85
        elif d <= 180:
            rel = 0.65
        elif d <= 360:
            rel = 0.45
        else:
            rel = 0.30
    else:
        rel = 0.55

    # Weak absolute priors on RP-C18 (only slight nudge)
    prior = 0.5
    nm = (name_hint or "").lower()
    if "RP" in (card.chromatography or "").upper() or "C18" in (card.chromatography or "").upper():
        polar = any(
            k in nm
            for k in (
                "sugar",
                "glucos",
                "phosphate",
                "sulfate",
                "amino acid",
                "carnitine",
                "glutathione",
            )
        )
        hydro = any(
            k in nm
            for k in ("lipid", "sterol", "cholic", "bile", "fatty", "triglycer", "ceramide")
        )
        if polar and seed_rt <= card.early_rt_max:
            prior = 0.7
        elif polar and seed_rt >= card.late_rt_min:
            prior = 0.35
        elif hydro and seed_rt >= card.late_rt_min * 0.5:
            prior = 0.7
        elif hydro and seed_rt <= card.early_rt_max:
            prior = 0.4

    return max(0.0, min(1.0, 0.65 * rel + 0.35 * prior))


def parse_rt_seconds(meta: dict[str, str] | None) -> float | None:
    if not meta:
        return None
    for key in ("RTINSECONDS", "RTINSECONDS.", "RETENTION_TIME", "RT", "RTs"):
        if key in meta:
            try:
                return float(str(meta[key]).split()[0])
            except ValueError:
                continue
    return None


def parse_ion_mode(meta: dict[str, str] | None) -> str | None:
    if not meta:
        return None
    for key in ("IONMODE", "ION_MODE", "POLARITY"):
        if key in meta:
            v = str(meta[key]).strip().upper()
            if v in {"P", "POS", "POSITIVE", "+"}:
                return "positive"
            if v in {"N", "NEG", "NEGATIVE", "-"}:
                return "negative"
            return v.lower()
    # CHARGE=1+ / 1- is a common MGF fallback
    ch = str(meta.get("CHARGE") or "").strip()
    if ch.endswith("-") or ch.startswith("-"):
        return "negative"
    if ch.endswith("+") or (ch and ch[0].isdigit() and "+" in ch):
        return "positive"
    return None


def resolve_method_polarity(
    *,
    seed_ion_mode: str | None = None,
    method: MethodCard | None = None,
    default: str = "both",
) -> str:
    """
    Prefer spectrum-level IONMODE over study-level method defaults.

    ASTRAL / multi-study packs mix ESI+ and ESI-; a single card polarity of
    ``positive`` misleads models when the seed MGF is Negative.
    """
    if seed_ion_mode in {"positive", "negative"}:
        return seed_ion_mode
    if method and method.polarity in {"positive", "negative", "both"}:
        return method.polarity
    return default
