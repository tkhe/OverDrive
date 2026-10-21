import logging
import os

from overdrive.utils.file_io import PathHandler
from overdrive.utils.file_io import PathManager


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

    IMAGENET_MODELS = {}

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


PathManager.register_handler(Detectron2PathHandler())
PathManager.register_handler(OpenMMLabPathHandler())
PathManager.register_handler(OverDrivePathHandler())
PathManager.register_handler(TIMMPathHandler())
PathManager.register_handler(ThirdPartyPathHandler())
PathManager.register_handler(TorchvisionPathHandler())
