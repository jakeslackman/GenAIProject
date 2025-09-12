import os
import sys

# Ensure project root is on sys.path so that `src` package can be imported
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.state.tx.models.state_transition import StateTransitionPerturbationModel
import anndata as ad
import scanpy as sc


adata = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/rpe1_normalized_singlecell_01.h5ad")

model = StateTransitionPerturbationModel.load_from_checkpoint("/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_state_sm_batch/rpe1/checkpoints/last.ckpt")

model.eval()

model.predict(adata)
