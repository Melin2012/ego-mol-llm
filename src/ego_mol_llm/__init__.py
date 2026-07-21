"""ego-mol-llm: structure prediction from molecular-network ego neighborhoods."""

from ego_mol_llm.predict import PredictionResult, predict_from_graphml
from ego_mol_llm.graphml import MolecularNetwork, load_graphml

__version__ = "0.1.0"
__all__ = [
    "MolecularNetwork",
    "load_graphml",
    "predict_from_graphml",
    "PredictionResult",
    "__version__",
]
