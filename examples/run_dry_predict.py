"""Example: blind dry-run prediction on a GraphML network."""

from pathlib import Path

from ego_mol_llm import predict_from_graphml
from ego_mol_llm.report import export_report

# Point this at your GraphML export
GRAPHML = Path(
    r"C:\Users\AlexeyMelnik\Downloads\HNSW_1-Methyl-1,2,3,4-tetrahydro-beta-carboline-3-carboxylic acid_AROMEC18COLGATE001635.graphml"
)
OUT = Path(__file__).resolve().parents[1] / "outputs" / "example_dry"

def main() -> None:
    if not GRAPHML.exists():
        raise SystemExit(f"GraphML not found: {GRAPHML}")

    result = predict_from_graphml(
        GRAPHML,
        backend="dry-run",
        hide_seed_name=True,
        max_neighbors=25,
        include_two_hop=True,
    )
    paths = export_report(result, OUT)
    print("Prediction:")
    for k, v in result.to_dict().items():
        if k in {"rationale", "alternatives"}:
            continue
        print(f"  {k}: {v}")
    print("Rationale:", result.prediction.rationale)
    print("Wrote:", paths)


if __name__ == "__main__":
    main()
