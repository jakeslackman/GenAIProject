from state.tx.models.state_transition import StateTransitionPerturbationModel
import anndata as ad


model = StateTransitionPerturbationModel.load_from_checkpoint("/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_state_sm_batch/rpe1/checkpoints/last.ckpt")

model.eval()

model.to("cuda")

model.predict(adata)