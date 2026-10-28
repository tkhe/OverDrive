from overdrive.config import LazyCall as L
from overdrive.model_zoo import get_config
from overdrive.modeling.distillation import ConfigurableDistiller
from overdrive.modeling.distillation import ModuleOutputsRecorder
from overdrive.modeling.loss import ChannelWiseDivergence
from overdrive.modeling.loss import PKDLoss
from overdrive.modeling.meta_arch import SingleTeacherDistillation

student = get_config("common/models/retinanet.py").model
teacher = get_config("COCO-Detection/retinanet/retinanet_resnet101_fpn_3x.py").model

model = L(SingleTeacherDistillation)(
    student=student,
    teacher=teacher,
    distiller=L(ConfigurableDistiller)(
        student_recorders=dict(
            fpn=L(ModuleOutputsRecorder)(source="neck"),
            cls_logits=L(ModuleOutputsRecorder)(source="box_head.cls_score"),
        ),
        teacher_recorders=dict(
            fpn=L(ModuleOutputsRecorder)(source="neck"),
            cls_logits=L(ModuleOutputsRecorder)(source="box_head.cls_score"),
        ),
        distill_losses=dict(
            loss_pkd_p3=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p4=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p5=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p6=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p7=L(PKDLoss)(loss_weight=1.0),
            loss_cwd_cls_p3=L(ChannelWiseDivergence)(tau=1.0, loss_weight=1.0),
            loss_cwd_cls_p4=L(ChannelWiseDivergence)(tau=1.0, loss_weight=1.0),
            loss_cwd_cls_p5=L(ChannelWiseDivergence)(tau=1.0, loss_weight=1.0),
            loss_cwd_cls_p6=L(ChannelWiseDivergence)(tau=1.0, loss_weight=1.0),
            loss_cwd_cls_p7=L(ChannelWiseDivergence)(tau=1.0, loss_weight=1.0),
        ),
        loss_forward_mappings=dict(
            loss_pkd_p3=dict(
                student=dict(recorder="fpn", from_student=True, data_idx="p3"),
                teacher=dict(recorder="fpn", from_student=False, data_idx="p3"),
            ),
            loss_pkd_p4=dict(
                student=dict(recorder="fpn", from_student=True, data_idx="p4"),
                teacher=dict(recorder="fpn", from_student=False, data_idx="p4"),
            ),
            loss_pkd_p5=dict(
                student=dict(recorder="fpn", from_student=True, data_idx="p5"),
                teacher=dict(recorder="fpn", from_student=False, data_idx="p5"),
            ),
            loss_pkd_p6=dict(
                student=dict(recorder="fpn", from_student=True, data_idx="p6"),
                teacher=dict(recorder="fpn", from_student=False, data_idx="p6"),
            ),
            loss_pkd_p7=dict(
                student=dict(recorder="fpn", from_student=True, data_idx="p7"),
                teacher=dict(recorder="fpn", from_student=False, data_idx="p7"),
            ),
            loss_cwd_cls_p3=dict(
                student=dict(recorder="cls_logits", from_student=True, record_idx=0),
                teacher=dict(recorder="cls_logits", from_student=False, record_idx=0),
            ),
            loss_cwd_cls_p4=dict(
                student=dict(recorder="cls_logits", from_student=True, record_idx=1),
                teacher=dict(recorder="cls_logits", from_student=False, record_idx=1),
            ),
            loss_cwd_cls_p5=dict(
                student=dict(recorder="cls_logits", from_student=True, record_idx=2),
                teacher=dict(recorder="cls_logits", from_student=False, record_idx=2),
            ),
            loss_cwd_cls_p6=dict(
                student=dict(recorder="cls_logits", from_student=True, record_idx=3),
                teacher=dict(recorder="cls_logits", from_student=False, record_idx=3),
            ),
            loss_cwd_cls_p7=dict(
                student=dict(recorder="cls_logits", from_student=True, record_idx=4),
                teacher=dict(recorder="cls_logits", from_student=False, record_idx=4),
            ),
        ),
    ),
)

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").SGD
optimizer.lr = 0.01

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x

train = get_config("common/train.py").train
train.init_checkpoint = dict(
    student="torchvision://ImageNet/ResNet-50",
    teacher="overdrive://COCO-Detection/RetinaNet/retinanet_resnet101_fpn_3x",
)
train.output_dir = "./output/retinanet_resnet50_fpn_retinanet_resnet101_fpn_1x_coco_detection"
