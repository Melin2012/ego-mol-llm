from pathlib import Path

from ego_mol_llm.paths import discover_graphml, make_run_dir
from ego_mol_llm.batch import run_batch


def test_make_run_dir_unique(tmp_path: Path):
    g = tmp_path / "demo_network.graphml"
    g.write_text("<graphml/>", encoding="utf-8")
    a = make_run_dir(parent=tmp_path / "runs", graphml=g, backend="dry-run")
    b = make_run_dir(parent=tmp_path / "runs", graphml=g, backend="dry-run")
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert a.parent == b.parent


def test_fixed_out(tmp_path: Path):
    fixed = tmp_path / "exact"
    out = make_run_dir(fixed=fixed)
    assert out == fixed.resolve()
    assert out.is_dir()


def test_discover_graphml(tmp_path: Path):
    d = tmp_path / "nets"
    d.mkdir()
    f1 = d / "a.graphml"
    f2 = d / "b.graphml"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")
    (d / "ignore.txt").write_text("z", encoding="utf-8")
    found = discover_graphml([d])
    assert len(found) == 2


def test_batch_dry_run_on_mtca_if_present(tmp_path: Path):
    mtca = Path(
        r"C:\Users\AlexeyMelnik\Downloads\HNSW_1-Methyl-1,2,3,4-tetrahydro-beta-carboline-3-carboxylic acid_AROMEC18COLGATE001635.graphml"
    )
    if not mtca.exists():
        return
    # minimal fake second copy for batch of 2 unique stems
    copy = tmp_path / "copy_mtca.graphml"
    copy.write_bytes(mtca.read_bytes())
    results, root = run_batch(
        [mtca, copy],
        backend="dry-run",
        model="dry",
        out_parent=tmp_path / "runs",
    )
    assert len(results) == 2
    assert any(r.ok for r in results), [r.error for r in results]
    assert (root / "batch_summary.csv").exists()
    assert (root / "batch_summary.md").exists()
