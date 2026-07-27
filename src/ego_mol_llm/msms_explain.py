"""
Fast offline MS/MS explanation for annotation propagation.

No SIRIUS / no web services. Designed to run on every sample:
  - labeled neutral losses (common metabolites / NP chemistry)
  - diagnostic ions beyond peptide immoniums
  - shared / unique peaks vs top spectral neighbors
  - Δm/z between seed and neighbor precursors with chemical guesses

SIRIUS/CSI remains optional for hard / high-value cases only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# (label, exact_mass_Da, chemistry_hint)
COMMON_LOSSES: list[tuple[str, float, str]] = [
    ("H2O", 18.0106, "OH / carboxylic acid / alcohol"),
    ("2H2O", 36.0211, "poly-OH / bile acid OH cascade"),
    ("NH3", 17.0265, "amine / amide / amino acid"),
    ("CO", 27.9949, "carbonyl / phenol / flavone"),
    ("CO2", 43.9898, "carboxylic acid / carbonate"),
    ("HCOOH", 46.0055, "carboxylic acid (formic)"),
    ("CH2O", 30.0106, "methoxy / sugar / phenol-OMe"),
    ("CH3OH", 32.0262, "methoxy"),
    ("C2H4", 28.0313, "ethyl / alkyl chain"),
    ("C2H2O", 42.0106, "acetyl / ketene"),
    ("C3H6", 42.0470, "propyl / acyl chain unit"),
    ("C4H8", 56.0626, "butyl / isovaleryl-related"),
    ("SO3", 79.9568, "sulfate conjugate"),
    ("H2SO4", 97.9674, "sulfate"),
    ("HPO3", 79.9663, "phosphate"),
    ("H3PO4", 97.9769, "phosphate ester"),
    ("C6H10O5", 162.0528, "hexose (−H2O from sugar)"),
    ("C6H12O6", 180.0634, "hexose"),
    ("C5H8O4", 132.0423, "pentose-related"),
    ("C2H5N", 43.0422, "aminoethyl / ethanolamine-ish"),
    ("C2H5NO2", 75.0320, "glycine"),
    ("C3H5NO2", 87.0320, "alanine / sarcosine-related"),
    ("C2H4O2", 60.0211, "acetic acid / acetate"),
    ("C5H9NO", 99.0684, "proline-related / acyl"),
    ("C9H8O2", 148.0524, "cinnamoyl-related"),
    ("C8H8O2", 136.0524, "anisole / methoxyphenyl"),
]

# (label, m/z, chemistry_hint) — positive-mode biased; still useful as soft clues
DIAGNOSTIC_IONS: list[tuple[str, float, str]] = [
    # amino acid immoniums / related
    ("Gly_immonium_30", 30.034, "glycine"),
    ("Val_immonium_72", 72.081, "Val"),
    ("Pro_immonium_70", 70.065, "Pro"),
    ("Leu_Ile_immonium_86", 86.097, "Leu/Ile"),
    ("His_immonium_110", 110.071, "His"),
    ("Phe_immonium_120", 120.081, "Phe"),
    ("Tyr_immonium_136", 136.076, "Tyr"),
    ("Trp_related_159", 159.092, "Trp"),
    ("Phe_related_166", 166.086, "Phe / acyl-Phe"),
    # acylglycines / conjugates
    ("protonated_glycine_76", 76.039, "acylglycine (Gly+H)"),
    ("acyl_C5H9O_85", 85.065, "C5 acyl fragment (isovaleryl-like)"),
    # bile acids / steroids (soft)
    ("BA_water_cascade", 355.26, "steroid/BA-ish (soft; check context)"),
    # carnitine
    ("carnitine_fragment_85", 85.029, "carnitine-related (also generic)"),
    ("TMA_60", 60.081, "choline/TMA-related"),
    # phenolics
    ("phenol_93", 93.034, "phenol"),
    ("catechol_109", 109.029, "catechol / dihydroxy"),
    ("trolox_like_165", 165.055, "phenolic acid-ish"),
    # sugars
    ("hexose_oxonium_163", 163.060, "hexose oxonium"),
    ("hexose_145", 145.050, "dehydrated hexose"),
]


@dataclass
class LabeledLoss:
    frag_mz: float
    intensity: float
    loss_da: float
    label: str
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticHit:
    label: str
    mz: float
    intensity: float
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeighborPeakDiff:
    neighbor_id: str
    neighbor_name: str | None
    msms_cosine: float | None
    delta_precursor_mz: float | None
    delta_guess: str
    n_shared_peaks: int
    shared_examples: list[str] = field(default_factory=list)
    seed_unique_examples: list[str] = field(default_factory=list)
    neighbor_unique_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MsmsExplanation:
    labeled_losses: list[LabeledLoss] = field(default_factory=list)
    diagnostics: list[DiagnosticHit] = field(default_factory=list)
    neighbor_diffs: list[NeighborPeakDiff] = field(default_factory=list)
    chemistry_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labeled_losses": [x.to_dict() for x in self.labeled_losses],
            "diagnostics": [x.to_dict() for x in self.diagnostics],
            "neighbor_diffs": [x.to_dict() for x in self.neighbor_diffs],
            "chemistry_hints": list(self.chemistry_hints),
        }


def label_loss(loss_da: float, tol: float = 0.02) -> tuple[str, str]:
    """Return (label, hint) for a precursor−fragment mass difference."""
    best: tuple[str, str, float] | None = None
    for lab, mass, hint in COMMON_LOSSES:
        d = abs(loss_da - mass)
        if d <= tol and (best is None or d < best[2]):
            best = (lab, hint, d)
    if best:
        return best[0], best[1]
    return "", ""


def explain_neutral_losses(
    peaks: list[tuple[float, float]],
    precursor_mz: float | None,
    *,
    top_n: int = 15,
    min_rel: float = 0.005,
) -> list[LabeledLoss]:
    if precursor_mz is None or not peaks:
        return []
    tops = sorted(peaks, key=lambda x: -x[1])
    base = tops[0][1] or 1.0
    out: list[LabeledLoss] = []
    for mz, inten in tops[: max(top_n * 4, 40)]:
        if inten / base < min_rel and len(out) >= 5:
            continue
        loss = float(precursor_mz) - float(mz)
        if loss < 4.5 or loss > 320:
            continue
        lab, hint = label_loss(loss)
        out.append(
            LabeledLoss(
                frag_mz=float(mz),
                intensity=float(inten),
                loss_da=loss,
                label=lab,
                hint=hint,
            )
        )
        if len(out) >= top_n:
            break
    return out


def explain_diagnostics(
    peaks: list[tuple[float, float]],
    *,
    tol: float = 0.02,
    min_rel: float = 0.01,
) -> list[DiagnosticHit]:
    if not peaks:
        return []
    base = max(i for _, i in peaks) or 1.0
    found: list[DiagnosticHit] = []
    for lab, target, hint in DIAGNOSTIC_IONS:
        best_mz, best_i = None, 0.0
        for mz, inten in peaks:
            if abs(mz - target) <= tol and inten > best_i:
                best_mz, best_i = mz, inten
        if best_mz is not None and best_i / base >= min_rel:
            found.append(
                DiagnosticHit(
                    label=lab,
                    mz=float(best_mz),
                    intensity=float(best_i),
                    hint=hint,
                )
            )
    found.sort(key=lambda x: -x.intensity)
    return found


def _peak_set(
    peaks: list[tuple[float, float]],
    *,
    top_n: int = 40,
    bin_da: float = 0.02,
) -> dict[int, tuple[float, float]]:
    """Bin top peaks: key = round(mz/bin), value = (mz, intensity)."""
    out: dict[int, tuple[float, float]] = {}
    for mz, inten in sorted(peaks, key=lambda x: -x[1])[:top_n]:
        k = int(round(mz / bin_da))
        prev = out.get(k)
        if prev is None or inten > prev[1]:
            out[k] = (float(mz), float(inten))
    return out


def guess_delta_chemistry(dmz: float | None) -> str:
    if dmz is None:
        return "unknown"
    d = abs(float(dmz))
    # signed note for direction
    sign = "seed heavier" if float(dmz) > 0 else "neighbor heavier"
    pairs = [
        (0.0, 0.02, "near-isobar (same ion / isotope)"),
        (1.003, 0.02, "13C isotope unit"),
        (2.016, 0.03, "H2 (unsaturation / reduction)"),
        (14.016, 0.03, "CH2 (homolog)"),
        (15.995, 0.03, "O (oxidation / epoxide)"),
        (18.011, 0.03, "H2O"),
        (28.031, 0.04, "C2H4 / CO"),
        (42.011, 0.04, "C2H2O (acetyl) or C3H6"),
        (42.047, 0.04, "C3H6 (propyl / isobaric)"),
        (44.026, 0.04, "C2H4O / CO2-ish"),
        (56.063, 0.05, "C4H8 (isovaleryl-related chain)"),
        (57.021, 0.05, "C2H3NO? / glycine-related Δ"),
        (75.032, 0.05, "glycine residue-ish"),
        (79.957, 0.05, "SO3"),
        (80.026, 0.05, "SO3 / HPO3 region"),
        (162.053, 0.06, "hexose (−H2O)"),
        (176.032, 0.06, "glucuronide-related"),
    ]
    for target, tol, lab in pairs:
        if abs(d - target) <= tol:
            return f"{lab} ({sign}, |Δ|={d:.4f})"
    if d < 0.5:
        return f"near-isobar |Δ|={d:.4f} ({sign})"
    return f"|Δm/z|={d:.4f} ({sign}; no common unit match)"


def compare_to_neighbor(
    seed_peaks: list[tuple[float, float]],
    neighbor_peaks: list[tuple[float, float]],
    *,
    neighbor_id: str,
    neighbor_name: str | None = None,
    msms_cosine: float | None = None,
    seed_mz: float | None = None,
    neighbor_mz: float | None = None,
    bin_da: float = 0.02,
    example_n: int = 5,
) -> NeighborPeakDiff:
    sset = _peak_set(seed_peaks, bin_da=bin_da)
    nset = _peak_set(neighbor_peaks, bin_da=bin_da)
    shared_keys = sorted(set(sset) & set(nset), key=lambda k: -sset[k][1])
    seed_only = sorted(set(sset) - set(nset), key=lambda k: -sset[k][1])
    neigh_only = sorted(set(nset) - set(sset), key=lambda k: -nset[k][1])

    def fmt(keys: list[int], src: dict[int, tuple[float, float]], n: int) -> list[str]:
        out = []
        base = max((v[1] for v in src.values()), default=1.0) or 1.0
        for k in keys[:n]:
            mz, inten = src[k]
            out.append(f"{mz:.4f} ({100*inten/base:.0f}%)")
        return out

    dmz = None
    if seed_mz is not None and neighbor_mz is not None:
        dmz = float(seed_mz) - float(neighbor_mz)

    return NeighborPeakDiff(
        neighbor_id=str(neighbor_id),
        neighbor_name=neighbor_name,
        msms_cosine=msms_cosine,
        delta_precursor_mz=dmz,
        delta_guess=guess_delta_chemistry(dmz),
        n_shared_peaks=len(shared_keys),
        shared_examples=fmt(shared_keys, sset, example_n),
        seed_unique_examples=fmt(seed_only, sset, example_n),
        neighbor_unique_examples=fmt(neigh_only, nset, example_n),
    )


def chemistry_hints_from_evidence(
    losses: list[LabeledLoss],
    diags: list[DiagnosticHit],
) -> list[str]:
    hints: list[str] = []
    labs = {x.label for x in losses if x.label}
    diags_l = {d.label for d in diags}

    if "H2O" in labs or "2H2O" in labs:
        hints.append("Water loss(es) → OH-rich (alcohol/phenol/carboxylic/BA).")
    if "CO2" in labs or "HCOOH" in labs:
        hints.append("CO2/HCOOH loss → carboxylic acid motif likely.")
    if "SO3" in labs or "H2SO4" in labs:
        hints.append("SO3/H2SO4 loss → sulfate conjugate candidate.")
    if "H3PO4" in labs or "HPO3" in labs:
        hints.append("Phosphate losses → phosphorylated metabolite.")
    if "C6H10O5" in labs or "C6H12O6" in labs or any("hexose" in d for d in diags_l):
        hints.append("Hexose-related loss/ion → glycoside.")
    if "protonated_glycine_76" in diags_l or "C2H5NO2" in labs:
        hints.append("m/z~76 (Gly+H) and/or Gly loss → acylglycine / glycine conjugate.")
    if "Phe_immonium_120" in diags_l or "Phe_related_166" in diags_l:
        hints.append("Phe immonium/related → phenylalanine moiety.")
    if "Leu_Ile_immonium_86" in diags_l and "protonated_glycine_76" in diags_l:
        hints.append("Leu/Ile immonium + Gly → possible branched acyl-glycine.")
    if "C4H8" in labs:
        hints.append("C4H8 loss → butyl/isovaleryl-type chain chemistry.")
    if not hints:
        hints.append(
            "No strong class diagnostics; rely on mass + high dual-cosine neighbors."
        )
    return hints


def build_msms_explanation(
    seed_peaks: list[tuple[float, float]],
    precursor_mz: float | None,
    *,
    neighbor_spectra: list[dict[str, Any]] | None = None,
    max_neighbors: int = 3,
) -> MsmsExplanation:
    """
    neighbor_spectra items:
      {id, name, peaks, mz, msms_cosine}
    """
    losses = explain_neutral_losses(seed_peaks, precursor_mz)
    diags = explain_diagnostics(seed_peaks)
    diffs: list[NeighborPeakDiff] = []
    for nb in (neighbor_spectra or [])[:max_neighbors]:
        peaks = nb.get("peaks") or []
        if not peaks:
            continue
        diffs.append(
            compare_to_neighbor(
                seed_peaks,
                peaks,
                neighbor_id=str(nb.get("id") or "?"),
                neighbor_name=nb.get("name"),
                msms_cosine=nb.get("msms_cosine"),
                seed_mz=precursor_mz,
                neighbor_mz=nb.get("mz"),
            )
        )
    hints = chemistry_hints_from_evidence(losses, diags)
    return MsmsExplanation(
        labeled_losses=losses,
        diagnostics=diags,
        neighbor_diffs=diffs,
        chemistry_hints=hints,
    )


def format_explanation_for_prompt(exp: MsmsExplanation, *, max_losses: int = 12) -> list[str]:
    lines = [
        "",
        "=== MS/MS EXPLANATION (offline, fast; not SIRIUS) ===",
    ]
    if exp.chemistry_hints:
        lines.append("  class hints:")
        for h in exp.chemistry_hints:
            lines.append(f"    - {h}")

    if exp.diagnostics:
        base = exp.diagnostics[0].intensity or 1.0
        bits = []
        for d in exp.diagnostics[:10]:
            bits.append(
                f"{d.label}@{d.mz:.3f} ({100*d.intensity/base:.0f}%: {d.hint})"
            )
        lines.append("  diagnostic ions: " + "; ".join(bits))

    if exp.labeled_losses:
        loss_bits = []
        for L in exp.labeled_losses[:max_losses]:
            if L.label:
                loss_bits.append(
                    f"{L.frag_mz:.2f} (−{L.loss_da:.2f} {L.label}"
                    f"{': ' + L.hint if L.hint else ''})"
                )
            else:
                loss_bits.append(f"{L.frag_mz:.2f} (−{L.loss_da:.2f})")
        lines.append("  labeled losses from precursor: " + "; ".join(loss_bits))

    if exp.neighbor_diffs:
        lines.append("  vs top MS/MS-similar neighbors:")
        for d in exp.neighbor_diffs:
            cos = f"{d.msms_cosine:.3f}" if d.msms_cosine is not None else "?"
            nm = d.neighbor_name or d.neighbor_id
            lines.append(
                f"    • {nm} | msms_cos={cos} | {d.delta_guess} | "
                f"shared_peaks≈{d.n_shared_peaks}"
            )
            if d.shared_examples:
                lines.append(f"      shared: {', '.join(d.shared_examples)}")
            if d.seed_unique_examples:
                lines.append(f"      seed-only: {', '.join(d.seed_unique_examples)}")
            if d.neighbor_unique_examples:
                lines.append(
                    f"      neighbor-only: {', '.join(d.neighbor_unique_examples)}"
                )
        lines.append(
            "  Use shared peaks as scaffold evidence; seed-only peaks + Δm/z "
            "to choose among mass-consistent homologs/conjugates."
        )
    return lines
