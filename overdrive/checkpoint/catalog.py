import logging
import os

from tabulate import tabulate

from overdrive.utils.file_io import PathHandler
from overdrive.utils.file_io import PathManager

__all__ = ["query_model_catalog"]


class Detectron2ModelCatalog:
    """
    Store mappings from names to detectron2 models.
    """

    PREFIX = "https://huggingface.co/tkhe/OverDrive/resolve/main/detectron2/"

    MODEL_GROUPS: tuple[str, ...] = ("ImageNet", "COCO-Detection", "COCO-InstanceSegmentation", "COCO-Keypoint")

    IMAGENET_MODELS = {}

    COCO_DETECTION_MODELS = {}

    COCO_INSTANCE_SEGMENTATION_MODELS = {}

    COCO_KEYPOINT_MODELS = {}

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return Detectron2ModelCatalog._get_imagenet_models(name)
        elif name.startswith("COCO-Detection/"):
            return Detectron2ModelCatalog._get_coco_detection_models(name)
        elif name.startswith("COCO-InstanceSegmentation/"):
            return Detectron2ModelCatalog._get_coco_instance_segmentation_models(name)
        elif name.startswith("COCO-Keypoint/"):
            return Detectron2ModelCatalog._get_coco_keypoint_models(name)
        else:
            raise RuntimeError(f"{name} not present in detectron2 catalog. Available model groups: {', '.join(Detectron2ModelCatalog.MODEL_GROUPS)}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("ImageNet/") :]
        try:
            path = Detectron2ModelCatalog.IMAGENET_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(Detectron2ModelCatalog.IMAGENET_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in detectron2://ImageNet catalog. Available models: {available_models}") from e
        return "".join([prefix, "ImageNet/", path])

    @staticmethod
    def _get_coco_detection_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("COCO-Detection/") :]
        try:
            path = Detectron2ModelCatalog.COCO_DETECTION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(Detectron2ModelCatalog.COCO_DETECTION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in detectron2://COCO-Detection catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-Detection/", path])

    @staticmethod
    def _get_coco_instance_segmentation_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("COCO-InstanceSegmentation/") :]
        try:
            path = Detectron2ModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(Detectron2ModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in detectron2://COCO-InstanceSegmentation catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-InstanceSegmentation/", path])

    @staticmethod
    def _get_coco_keypoint_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("COCO-Keypoint/") :]
        try:
            path = Detectron2ModelCatalog.COCO_KEYPOINT_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(Detectron2ModelCatalog.COCO_KEYPOINT_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in detectron2://COCO-Keypoint catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-Keypoint/", path])


class OpenMMLabModelCatalog:
    """
    Store mappings from names to openmmlab models
    """

    PREFIX = "https://huggingface.co/tkhe/OverDrive/resolve/main/openmmlab/"

    MODEL_GROUPS: tuple[str, ...] = ("mmpretrain", "mmdet", "mmyolo", "mmseg", "mmpose")

    MMPRETRAIN_MODELS = {}

    MMDET_MODELS = {}

    MMYOLO_MODELS = {}

    MMSEG_MODELS = {}

    MMPOSE_MODELS = {}

    @staticmethod
    def get(name: str):
        if name.startswith("mmpretrain/"):
            return OpenMMLabModelCatalog._get_mmpretrain_models(name)
        elif name.startswith("mmdet/"):
            return OpenMMLabModelCatalog._get_mmdet_models(name)
        elif name.startswith("mmyolo/"):
            return OpenMMLabModelCatalog._get_mmyolo_models(name)
        elif name.startswith("mmseg/"):
            return OpenMMLabModelCatalog._get_mmseg_models(name)
        elif name.startswith("mmpose/"):
            return OpenMMLabModelCatalog._get_mmpose_models(name)
        else:
            raise RuntimeError(f"{name} not present in openmmlab catalog. Available model groups: {', '.join(OpenMMLabModelCatalog.MODEL_GROUPS)}")

    @staticmethod
    def _get_mmpretrain_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmpretrain/") :]
        try:
            path = OpenMMLabModelCatalog.MMPRETRAIN_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OpenMMLabModelCatalog.MMPRETRAIN_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in openmmlab://mmpretrain catalog. Available models: {available_models}") from e
        return "".join([prefix, "mmpretrain/", path])

    @staticmethod
    def _get_mmdet_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmdet/") :]
        try:
            path = OpenMMLabModelCatalog.MMDET_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OpenMMLabModelCatalog.MMDET_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in openmmlab://mmdet catalog. Available models: {available_models}") from e
        return "".join([prefix, "mmdet/", path])

    @staticmethod
    def _get_mmyolo_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmyolo/") :]
        try:
            path = OpenMMLabModelCatalog.MMYOLO_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OpenMMLabModelCatalog.MMYOLO_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in openmmlab://mmyolo catalog. Available models: {available_models}") from e
        return "".join([prefix, "mmyolo/", path])

    @staticmethod
    def _get_mmseg_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmseg/") :]
        try:
            path = OpenMMLabModelCatalog.MMSEG_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OpenMMLabModelCatalog.MMSEG_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in openmmlab://mmseg catalog. Available models: {available_models}") from e
        return "".join([prefix, "mmseg/", path])

    @staticmethod
    def _get_mmpose_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmpose/") :]
        try:
            path = OpenMMLabModelCatalog.MMPOSE_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OpenMMLabModelCatalog.MMPOSE_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in openmmlab://mmpose catalog. Available models: {available_models}") from e
        return "".join([prefix, "mmpose/", path])


class OverDriveModelCatalog:
    """
    Store mappings from names to OverDrive models.
    """

    PREFIX = "https://huggingface.co/tkhe/OverDrive/resolve/main/overdrive/"

    MODEL_GROUPS: tuple[str, ...] = (
        "ImageNet",
        "COCO-Detection",
        "COCO-InstanceSegmentation",
        "COCO-Keypoint",
        "ADE20K-SemanticSegmentation",
        "VisDrone-Detection",
        "PascalVOC-Detection",
        "PascalVOC-SemanticSegmentation",
    )

    IMAGENET_MODELS = {}

    COCO_DETECTION_MODELS = {}

    COCO_INSTANCE_SEGMENTATION_MODELS = {}

    COCO_KEYPOINT_MODELS = {}

    ADE20K_SEMANTIC_SEGMENTATION_MODELS = {}

    VISDRONE_DETECTION_MODELS = {}

    PASCALVOC_DETECTION_MODELS = {}

    PASCALVOC_SEMANTIC_SEGMENTATION_MODELS = {}

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return OverDriveModelCatalog._get_imagenet_models(name)
        elif name.startswith("COCO-Detection/"):
            return OverDriveModelCatalog._get_coco_detection_models(name)
        elif name.startswith("COCO-InstanceSegmentation/"):
            return OverDriveModelCatalog._get_coco_instance_segmentation_models(name)
        elif name.startswith("COCO-Keypoint/"):
            return OverDriveModelCatalog._get_coco_keypoint_models(name)
        elif name.startswith("ADE20K-SemanticSegmentation/"):
            return OverDriveModelCatalog._get_ade20k_semantic_segmentation_models(name)
        elif name.startswith("VisDrone-Detection/"):
            return OverDriveModelCatalog._get_visdrone_detection_models(name)
        elif name.startswith("PascalVOC-Detection/"):
            return OverDriveModelCatalog._get_pascal_voc_detection_models(name)
        elif name.startswith("PascalVOC-SemanticSegmentation/"):
            return OverDriveModelCatalog._get_pascal_voc_semantic_segmentation_models(name)
        else:
            raise RuntimeError(f"{name} not present in overdrive catalog. Available model groups: {', '.join(OverDriveModelCatalog.MODEL_GROUPS)}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("ImageNet/") :]
        try:
            path = OverDriveModelCatalog.IMAGENET_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.IMAGENET_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://ImageNet catalog. Available models: {available_models}") from e
        return "".join([prefix, "ImageNet/", path])

    @staticmethod
    def _get_coco_detection_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("COCO-Detection/") :]
        try:
            path = OverDriveModelCatalog.COCO_DETECTION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.COCO_DETECTION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://COCO-Detection catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-Detection/", path])

    @staticmethod
    def _get_coco_instance_segmentation_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("COCO-InstanceSegmentation/") :]
        try:
            path = OverDriveModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://COCO-InstanceSegmentation catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-InstanceSegmentation/", path])

    @staticmethod
    def _get_coco_keypoint_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("COCO-Keypoint/") :]
        try:
            path = OverDriveModelCatalog.COCO_KEYPOINT_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.COCO_KEYPOINT_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://COCO-Keypoint catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-Keypoint/", path])

    @staticmethod
    def _get_ade20k_semantic_segmentation_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("ADE20K-SemanticSegmentation/") :]
        try:
            path = OverDriveModelCatalog.ADE20K_SEMANTIC_SEGMENTATION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.ADE20K_SEMANTIC_SEGMENTATION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://ADE20K-SemanticSegmentation catalog. Available models: {available_models}") from e
        return "".join([prefix, "ADE20K-SemanticSegmentation/", path])

    @staticmethod
    def _get_visdrone_detection_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("VisDrone-Detection/") :]
        try:
            path = OverDriveModelCatalog.VISDRONE_DETECTION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.VISDRONE_DETECTION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://VisDrone-Detection catalog. Available models: {available_models}") from e
        return "".join([prefix, "VisDrone-Detection/", path])

    @staticmethod
    def _get_pascal_voc_detection_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("PascalVOC-Detection/") :]
        try:
            path = OverDriveModelCatalog.PASCALVOC_DETECTION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.PASCALVOC_DETECTION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://PascalVOC-Detection catalog. Available models: {available_models}") from e
        return "".join([prefix, "PascalVOC-Detection/", path])

    @staticmethod
    def _get_pascal_voc_semantic_segmentation_models(name: str):
        prefix = OverDriveModelCatalog.PREFIX
        model_name = name[len("PascalVOC-SemanticSegmentation/") :]
        try:
            path = OverDriveModelCatalog.PASCALVOC_SEMANTIC_SEGMENTATION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(OverDriveModelCatalog.PASCALVOC_SEMANTIC_SEGMENTATION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in overdrive://PascalVOC-SemanticSegmentation catalog. Available models: {available_models}") from e
        return "".join([prefix, "PascalVOC-SemanticSegmentation/", path])


class TIMMModelCatalog:
    """
    Store mappings from names to timm models.
    """

    PREFIX = "https://huggingface.co/tkhe/OverDrive/resolve/main/timm/"

    MODEL_GROUPS: tuple[str, ...] = ("ImageNet",)

    IMAGENET_MODELS = {
        "MobileNet-V1-x1.0": "MobileNetV1/mobilenetv1_x1_0-9ce2f8a7.safetensors",
        "GhostNet-V1-x1.0": "GhostNetV1/ghostnetv1_x1_0-635ef344.safetensors",
        "GhostNet-V2-x1.0": "GhostNetV2/ghostnetv2_x1_0-260038e5.safetensors",
        "GhostNet-V2-x1.3": "GhostNetV2/ghostnetv2_x1_3-0fc798d4.safetensors",
        "GhostNet-V2-x1.6": "GhostNetV2/ghostnetv2_x1_6-7ee596ed.safetensors",
    }

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return TIMMModelCatalog._get_imagenet_models(name)
        else:
            raise RuntimeError(f"{name} not present in timm catalog. Available model groups: {', '.join(TIMMModelCatalog.MODEL_GROUPS)}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = TIMMModelCatalog.PREFIX
        model_name = name[len("ImageNet/") :]
        try:
            path = TIMMModelCatalog.IMAGENET_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(TIMMModelCatalog.IMAGENET_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in timm://ImageNet catalog. Available models: {available_models}") from e
        return "".join([prefix, "ImageNet/", path])


class ThirdPartyModelCatalog:
    """
    Store mappings from names to third party models
    """

    PREFIX = "https://huggingface.co/tkhe/OverDrive/resolve/main/third_party/"

    MODEL_GROUPS: tuple[str, ...] = ("ImageNet", "COCO-Detection", "COCO-InstanceSegmentation")

    IMAGENET_MODELS = {}

    COCO_DETECTION_MODELS = {}

    COCO_INSTANCE_SEGMENTATION_MODELS = {}

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return ThirdPartyModelCatalog._get_imagenet_models(name)
        elif name.startswith("COCO-Detection/"):
            return ThirdPartyModelCatalog._get_coco_detection_models(name)
        elif name.startswith("COCO-InstanceSegmentation/"):
            return ThirdPartyModelCatalog._get_coco_instance_segmentation_models(name)
        else:
            raise RuntimeError(f"{name} not present in third_party catalog. Available model groups: {', '.join(ThirdPartyModelCatalog.MODEL_GROUPS)}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = ThirdPartyModelCatalog.PREFIX
        model_name = name[len("ImageNet/") :]
        try:
            path = ThirdPartyModelCatalog.IMAGENET_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(ThirdPartyModelCatalog.IMAGENET_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in third_party://ImageNet catalog. Available models: {available_models}") from e
        return "".join([prefix, "ImageNet/", path])

    @staticmethod
    def _get_coco_detection_models(name: str):
        prefix = ThirdPartyModelCatalog.PREFIX
        model_name = name[len("COCO-Detection/") :]
        try:
            path = ThirdPartyModelCatalog.COCO_DETECTION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(ThirdPartyModelCatalog.COCO_DETECTION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in third_party://COCO-Detection catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-Detection/", path])

    @staticmethod
    def _get_coco_instance_segmentation_models(name: str):
        prefix = ThirdPartyModelCatalog.PREFIX
        model_name = name[len("COCO-InstanceSegmentation/") :]
        try:
            path = ThirdPartyModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(ThirdPartyModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in third_party://COCO-InstanceSegmentation catalog. Available models: {available_models}") from e
        return "".join([prefix, "COCO-InstanceSegmentation/", path])


class TorchvisionModelCatalog:
    """
    Store mappings from names to torchvision models.
    """

    PREFIX = "https://huggingface.co/tkhe/OverDrive/resolve/main/torchvision/"

    MODEL_GROUPS: tuple[str, ...] = ("ImageNet",)

    IMAGENET_MODELS = {
        "ResNet-18": "ResNet/resnet18-64a3c2a6.safetensors",
        "ResNet-34": "ResNet/resnet34-848f9681.safetensors",
        "ResNet-50": "ResNet/resnet50-4bd85ca6.safetensors",
        "ResNet-101": "ResNet/resnet101-dba31703.safetensors",
        "ResNet-152": "ResNet/resnet152-4451f4a3.safetensors",
        "ResNeXt-50-32x4d": "ResNeXt/resnext50_32x4d-0dde6f4b.safetensors",
        "ResNeXt-101-32x8d": "ResNeXt/resnext101_32x8d-b2bfc121.safetensors",
        "ResNeXt-101-64x4d": "ResNeXt/resnext101_64x4d-eb74b497.safetensors",
        "MobileNet-V2-x1.0": "MobileNetV2/mobilenetv2_x1_0-4dd72eb8.safetensors",
        "MobileNet-V3-Large-x1.0": "MobileNetV3/mobilenetv3_large_x1_0-f2fa066e.safetensors",
        "MobileNet-V3-Small-x1.0": "MobileNetV3/mobilenetv3_small_x1_0-f4348bc9.safetensors",
        "ShuffleNet-V2-x0.5": "ShuffleNetV2/shufflenetv2_x0_5-de9aa40e.safetensors",
        "ShuffleNet-V2-x1.0": "ShuffleNetV2/shufflenetv2_x1_0-8fc1719d.safetensors",
        "ShuffleNet-V2-x1.5": "ShuffleNetV2/shufflenetv2_x1_5-551a2f40.safetensors",
        "ShuffleNet-V2-x2.0": "ShuffleNetV2/shufflenetv2_x2_0-4b7b1e83.safetensors",
    }

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return TorchvisionModelCatalog._get_imagenet_models(name)
        else:
            raise RuntimeError(f"{name} not present in torchvision catalog. Available model groups: {', '.join(TorchvisionModelCatalog.MODEL_GROUPS)}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = TorchvisionModelCatalog.PREFIX
        model_name = name[len("ImageNet/") :]
        try:
            path = TorchvisionModelCatalog.IMAGENET_MODELS[model_name]
        except KeyError as e:
            available_models = ", ".join(sorted(TorchvisionModelCatalog.IMAGENET_MODELS)) or "(none)"
            raise RuntimeError(f"{model_name} not present in torchvision://ImageNet catalog. Available models: {available_models}") from e
        return "".join([prefix, "ImageNet/", path])


class Detectron2PathHandler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"

    def get_supported_prefixes(self) -> list[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, force: bool = False, cache_dir: str = None, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)
        name = path[len(self.PREFIX) :]
        catalog_path = Detectron2ModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        relative_path = catalog_path[len(Detectron2ModelCatalog.PREFIX) :]
        local_dir = os.path.join("detectron2", os.path.dirname(relative_path))
        return PathManager.get_local_path(catalog_path, force=force, cache_dir=cache_dir, local_dir=local_dir, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class OpenMMLabPathHandler(PathHandler):
    """
    Resolve anything that's hosted under openmmlab's namespace.
    """

    PREFIX = "openmmlab://"

    def get_supported_prefixes(self) -> list[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, force: bool = False, cache_dir: str = None, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)
        name = path[len(self.PREFIX) :]
        catalog_path = OpenMMLabModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        relative_path = catalog_path[len(OpenMMLabModelCatalog.PREFIX) :]
        local_dir = os.path.join("openmmlab", os.path.dirname(relative_path))
        return PathManager.get_local_path(catalog_path, force=force, cache_dir=cache_dir, local_dir=local_dir, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class OverDrivePathHandler(PathHandler):
    """
    Resolve anything that's hosted under OverDrive's namespace.
    """

    PREFIX = "overdrive://"

    def get_supported_prefixes(self) -> list[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, force: bool = False, cache_dir: str = None, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)
        name = path[len(self.PREFIX) :]
        catalog_path = OverDriveModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        relative_path = catalog_path[len(OverDriveModelCatalog.PREFIX) :]
        local_dir = os.path.join("overdrive", os.path.dirname(relative_path))
        return PathManager.get_local_path(catalog_path, force=force, cache_dir=cache_dir, local_dir=local_dir, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class TIMMPathHandler(PathHandler):
    """
    Resolve anything that's hosted under timm's namespace.
    """

    PREFIX = "timm://"

    def get_supported_prefixes(self) -> list[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, force: bool = False, cache_dir: str = None, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)
        name = path[len(self.PREFIX) :]
        catalog_path = TIMMModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        relative_path = catalog_path[len(TIMMModelCatalog.PREFIX) :]
        local_dir = os.path.join("timm", os.path.dirname(relative_path))
        return PathManager.get_local_path(catalog_path, force=force, cache_dir=cache_dir, local_dir=local_dir, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class ThirdPartyPathHandler(PathHandler):
    """
    Resolve anything that's hosted under third_party's namespace.
    """

    PREFIX = "third_party://"

    def get_supported_prefixes(self) -> list[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, force: bool = False, cache_dir: str = None, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)
        name = path[len(self.PREFIX) :]
        catalog_path = ThirdPartyModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        relative_path = catalog_path[len(ThirdPartyModelCatalog.PREFIX) :]
        local_dir = os.path.join("third_party", os.path.dirname(relative_path))
        return PathManager.get_local_path(catalog_path, force=force, cache_dir=cache_dir, local_dir=local_dir, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class TorchvisionPathHandler(PathHandler):
    """
    Resolve anything that's hosted under torchvision's namespace.
    """

    PREFIX = "torchvision://"

    def get_supported_prefixes(self) -> list[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, force: bool = False, cache_dir: str = None, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        self._check_kwargs(kwargs)
        name = path[len(self.PREFIX) :]
        catalog_path = TorchvisionModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        relative_path = catalog_path[len(TorchvisionModelCatalog.PREFIX) :]
        local_dir = os.path.join("torchvision", os.path.dirname(relative_path))
        return PathManager.get_local_path(catalog_path, force=force, cache_dir=cache_dir, local_dir=local_dir, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


_MODEL_CATALOGS = {
    "detectron2": (
        Detectron2ModelCatalog,
        {
            "ImageNet": "IMAGENET_MODELS",
            "COCO-Detection": "COCO_DETECTION_MODELS",
            "COCO-InstanceSegmentation": "COCO_INSTANCE_SEGMENTATION_MODELS",
            "COCO-Keypoint": "COCO_KEYPOINT_MODELS",
        },
    ),
    "openmmlab": (
        OpenMMLabModelCatalog,
        {
            "mmpretrain": "MMPRETRAIN_MODELS",
            "mmdet": "MMDET_MODELS",
            "mmyolo": "MMYOLO_MODELS",
            "mmseg": "MMSEG_MODELS",
            "mmpose": "MMPOSE_MODELS",
        },
    ),
    "overdrive": (
        OverDriveModelCatalog,
        {
            "ImageNet": "IMAGENET_MODELS",
            "COCO-Detection": "COCO_DETECTION_MODELS",
            "COCO-InstanceSegmentation": "COCO_INSTANCE_SEGMENTATION_MODELS",
            "COCO-Keypoint": "COCO_KEYPOINT_MODELS",
            "ADE20K-SemanticSegmentation": "ADE20K_SEMANTIC_SEGMENTATION_MODELS",
            "VisDrone-Detection": "VISDRONE_DETECTION_MODELS",
            "PascalVOC-Detection": "PASCALVOC_DETECTION_MODELS",
            "PascalVOC-SemanticSegmentation": "PASCALVOC_SEMANTIC_SEGMENTATION_MODELS",
        },
    ),
    "timm": (
        TIMMModelCatalog,
        {
            "ImageNet": "IMAGENET_MODELS",
        },
    ),
    "third_party": (
        ThirdPartyModelCatalog,
        {
            "ImageNet": "IMAGENET_MODELS",
            "COCO-Detection": "COCO_DETECTION_MODELS",
            "COCO-InstanceSegmentation": "COCO_INSTANCE_SEGMENTATION_MODELS",
        },
    ),
    "torchvision": (
        TorchvisionModelCatalog,
        {
            "ImageNet": "IMAGENET_MODELS",
        },
    ),
}


def query_model_catalog(
    name: str | None = None,
    *,
    catalog: str | None = None,
    group: str | None = None,
    display: bool = True,
) -> list[dict[str, str]]:
    """
    Query locally registered model weights without downloading them.

    Args:
        name: Case-insensitive substring of the model name. Leading and trailing
            whitespace is ignored; None or an empty string matches all names.
            For example, "ResNet" matches "ResNet-50", but not "ResNeXt-50-32x4d".
        catalog: Exact, case-sensitive catalog name, such as "timm" or
            "torchvision". None searches all catalogs.
        group: Exact, case-sensitive group name, such as "ImageNet",
            "COCO-Detection", or "mmdet". None searches all groups.
        display: Print a tabulate table and match count when True.

    Returns:
        Records with ``catalog``, ``group``, ``model``, and ``uri`` string fields,
        in dictionary insertion order: catalogs, then groups within each catalog,
        then models within each group. Filtering preserves this order. All filters
        must match. Entries with the same model name in different catalogs or
        groups are retained.
        URIs use the existing loading schemes, e.g.
        ``torchvision://ImageNet/ResNet-50``. No matches returns an empty list.

    Raises:
        ValueError: If catalog is unknown or group is unknown across all catalogs.
            A known group absent from the selected catalog returns no matches.

    Examples:
        >>> weights = query_model_catalog()
        >>> weights = query_model_catalog(catalog="timm")
        >>> weights = query_model_catalog(group="ImageNet")
        >>> weights = query_model_catalog("ResNet")
        >>> weights = query_model_catalog("ResNet", catalog="torchvision")
        >>> weights = query_model_catalog("ResNet", group="COCO-Detection")
        >>> weights = query_model_catalog("ResNet", display=False)
    """
    if catalog is not None and catalog not in _MODEL_CATALOGS:
        raise ValueError(f"Unknown catalog {catalog!r}. Available catalogs: {', '.join(sorted(_MODEL_CATALOGS))}")

    available_groups = {key for _, groups in _MODEL_CATALOGS.values() for key in groups}
    if group is not None and group not in available_groups:
        raise ValueError(f"Unknown group {group!r}. Available groups: {', '.join(sorted(available_groups))}")

    query = name.strip().casefold() if name is not None else ""
    records = []
    for catalog_name, (catalog_class, groups) in _MODEL_CATALOGS.items():
        if catalog is not None and catalog_name != catalog:
            continue
        for group_name, attribute in groups.items():
            if group is not None and group_name != group:
                continue
            for model_name in getattr(catalog_class, attribute):
                if query in model_name.casefold():
                    records.append({
                        "catalog": catalog_name,
                        "group": group_name,
                        "model": model_name,
                        "uri": f"{catalog_name}://{group_name}/{model_name}",
                    })

    if display:
        rows = [(record["catalog"], record["group"], record["model"], record["uri"]) for record in records]
        print(tabulate(rows, headers=["Catalog", "Group", "Model", "URI"], tablefmt="pipe"))
        print(f"Matched weights: {len(records)}")
        if not records:
            print("No matching weights found.")

    return records


PathManager.register_handler(Detectron2PathHandler())
PathManager.register_handler(OpenMMLabPathHandler())
PathManager.register_handler(OverDrivePathHandler())
PathManager.register_handler(TIMMPathHandler())
PathManager.register_handler(ThirdPartyPathHandler())
PathManager.register_handler(TorchvisionPathHandler())
