from .retinanet_res18_res50_res101_1x import dataloader
from .retinanet_res18_res50_res101_1x import lr_multipliers
from .retinanet_res18_res50_res101_1x import model
from .retinanet_res18_res50_res101_1x import optimizers
from .retinanet_res18_res50_res101_1x import train

# res18 learns res50's logits and res101's FPN features. res50 learns only
# res18's logits. res101 still trains on GT and supplies targets to res18.
edges = {"res18": {"res50": "loss_cwd_", "res101": "loss_pkd_"}, "res50": {"res18": "loss_cwd_"}}
model.distiller.distill_losses = {
    name: {
        peer: {key: value for key, value in model.distiller.distill_losses[name][peer].items() if key.startswith(prefix)}
        for peer, prefix in peers.items()
    }
    for name, peers in edges.items()
}
model.distiller.loss_forward_mappings = {
    name: {
        peer: {key: value for key, value in model.distiller.loss_forward_mappings[name][peer].items() if key.startswith(prefix)}
        for peer, prefix in peers.items()
    }
    for name, peers in edges.items()
}
model.distiller.recorders = {
    "res18": model.distiller.recorders.res18,
    "res50": dict(logits=model.distiller.recorders.res50.logits),
    "res101": dict(fpn=model.distiller.recorders.res101.fpn),
}
train.eval.eval_student = ["res18", "res50"]
optimizers.res50.lr = 0.005
train.output_dir = "./output/dml_retinanet_res18_res50_res101_sparse_1x"
