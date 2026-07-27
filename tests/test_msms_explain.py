"""Offline MS/MS explanation tests (no SIRIUS)."""

from __future__ import annotations

from ego_mol_llm.msms_explain import (
    build_msms_explanation,
    format_explanation_for_prompt,
    guess_delta_chemistry,
    label_loss,
)


def test_label_common_losses():
    lab, hint = label_loss(18.0106)
    assert lab == "H2O"
    assert "OH" in hint or "alcohol" in hint.lower() or "acid" in hint.lower()
    lab2, _ = label_loss(75.032)
    assert lab2 == "C2H5NO2"  # glycine


def test_acylglycine_like_spectrum():
    # synthetic-ish isovalerylglycine peaks
    precursor = 160.0968
    peaks = [
        (76.0394, 100.0),
        (57.0698, 96.0),
        (85.0647, 84.0),
        (142.0860, 5.0),  # -H2O
        (114.0914, 3.0),  # ~HCOOH
    ]
    exp = build_msms_explanation(peaks, precursor)
    labels = {d.label for d in exp.diagnostics}
    assert "protonated_glycine_76" in labels or any("76" in d.label for d in exp.diagnostics)
    loss_labs = {L.label for L in exp.labeled_losses if L.label}
    assert "H2O" in loss_labs or "HCOOH" in loss_labs
    assert any("glycine" in h.lower() or "acyl" in h.lower() for h in exp.chemistry_hints)


def test_neighbor_diff_and_prompt():
    seed = [(100.0, 100.0), (120.0, 50.0), (80.0, 20.0)]
    nb = [(100.0, 90.0), (120.0, 40.0), (140.0, 30.0)]
    exp = build_msms_explanation(
        seed,
        200.0,
        neighbor_spectra=[
            {
                "id": "1",
                "name": "neighborA",
                "peaks": nb,
                "mz": 186.0,  # Δ ~14 CH2
                "msms_cosine": 0.85,
            }
        ],
    )
    assert exp.neighbor_diffs
    assert exp.neighbor_diffs[0].n_shared_peaks >= 2
    assert "CH2" in exp.neighbor_diffs[0].delta_guess or "homolog" in exp.neighbor_diffs[0].delta_guess
    lines = format_explanation_for_prompt(exp)
    assert any("MS/MS EXPLANATION" in x for x in lines)
    assert any("shared" in x for x in lines)


def test_guess_delta():
    assert "isobar" in guess_delta_chemistry(0.001).lower() or "near" in guess_delta_chemistry(0.001).lower()
    g = guess_delta_chemistry(14.015)
    assert "CH2" in g or "homolog" in g
