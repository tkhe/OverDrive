from overdrive.config import LazyCall as L
from overdrive.model_zoo import get_config
from overdrive.modeling.distillation import DMLDistiller
from overdrive.modeling.distillation import ModuleOutputsRecorder
from overdrive.modeling.loss import ChannelWiseDivergence
from overdrive.modeling.loss import PKDLoss
from overdrive.modeling.meta_arch import DML

students = {
    "res18": get_config("COCO-Detection/retinanet/retinanet_resnet18_fpn_3x.py").model,
    "res50": get_config("common/models/retinanet.py").model,
    "res101": get_config("COCO-Detection/retinanet/retinanet_resnet101_fpn_3x.py").model,
}


def peer_losses():
    return {
        **{f"loss_pkd_p{level}": L(PKDLoss)(loss_weight=1.0) for level in range(3, 8)},
        **{f"loss_cwd_p{level}": L(ChannelWiseDivergence)(tau=1.0, loss_weight=1.0) for level in range(3, 8)},
    }


def peer_mappings():
    mappings = {}
    for level in range(3, 8):
        mappings[f"loss_pkd_p{level}"] = dict(
            student=dict(recorder="fpn", from_student=True, data_idx=f"p{level}"),
            teacher=dict(recorder="fpn", from_student=False, data_idx=f"p{level}"),
        )
        mappings[f"loss_cwd_p{level}"] = dict(
            student=dict(recorder="logits", from_student=True, record_idx=level - 3),
            teacher=dict(recorder="logits", from_student=False, record_idx=level - 3),
        )
    return mappings


model = L(DML)(
    models=students,
    distiller=L(DMLDistiller)(
        recorders={
            name: dict(fpn=L(ModuleOutputsRecorder)(source="neck"), logits=L(ModuleOutputsRecorder)(source="box_head.cls_score"))
            for name in students
        },
        distill_losses={name: {peer: peer_losses() for peer in students if peer != name} for name in students},
        loss_forward_mappings={name: {peer: peer_mappings() for peer in students if peer != name} for name in students},
    ),
)

dataloader = get_config("common/data/coco_detection.py").dataloader
optimizers = {name: get_config("common/optim.py").SGD for name in students}
for optimizer in optimizers.values():
    optimizer.lr = 0.01
lr_multipliers = {name: get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x for name in students}

train = get_config("common/train.py").train
train.eval = dict(period=1)  # eval_student may be a string, a list, or omitted.
train.best_checkpointer = dict(val_metric="bbox/AP", mode="max")
train.custom_hooks = []
train.init_checkpoint = dict(
    resume="",
    models={
        "res18": "torchvision://ImageNet/ResNet-18",
        "res50": "torchvision://ImageNet/ResNet-50",
        "res101": "torchvision://ImageNet/ResNet-101",
    },
)
train.gradient_clipping = dict(res18=dict(enable=True, type="norm", value=1.0, norm=2.0))
train.output_dir = "./output/dml_retinanet_res18_res50_res101_1x"
