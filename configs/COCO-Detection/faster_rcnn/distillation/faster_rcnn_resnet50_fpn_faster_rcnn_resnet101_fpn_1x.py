from overdrive.config import LazyCall as L
from overdrive.model_zoo import get_config
from overdrive.modeling.distillation import ConfigurableDistiller
from overdrive.modeling.distillation import ModuleOutputsRecorder
from overdrive.modeling.loss import PKDLoss
from overdrive.modeling.meta_arch import FPNTeacherDistillation

student = get_config("common/models/faster_rcnn.py").model
teacher = get_config("COCO-Detection/faster_rcnn/faster_rcnn_resnet101_fpn_3x.py").model

model = L(FPNTeacherDistillation)(
    student=student,
    teacher=teacher,
    distiller=L(ConfigurableDistiller)(
        student_recorders=dict(fpn=L(ModuleOutputsRecorder)(source="neck")),
        teacher_recorders=dict(fpn=L(ModuleOutputsRecorder)(source="neck")),
        distill_losses=dict(
            loss_pkd_p2=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p3=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p4=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p5=L(PKDLoss)(loss_weight=1.0),
            loss_pkd_p6=L(PKDLoss)(loss_weight=1.0),
        ),
        loss_forward_mappings=dict(
            loss_pkd_p2=dict(
                student=dict(recorder="fpn", from_student=True, data_idx="p2"),
                teacher=dict(recorder="fpn", from_student=False, data_idx="p2"),
            ),
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
        ),
    ),
)

dataloader = get_config("common/data/coco_detection.py").dataloader

optimizer = get_config("common/optim.py").SGD

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_1x

train = get_config("common/train.py").train
train.init_checkpoint = dict(
    student="torchvision://ImageNet/ResNet-50",
    teacher="overdrive://COCO-Detection/FasterRCNN/faster_rcnn_resnet101_fpn_3x",
)
train.output_dir = "./output/faster_rcnn_resnet50_fpn_faster_rcnn_resnet101_fpn_1x_coco_detection"
