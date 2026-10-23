import os

from overdrive.checkpoint import Checkpointer
from overdrive.config import LazyConfig
from overdrive.data import MetadataCatalog
from overdrive.engine import Trainer as DefaultTrainer
from overdrive.engine import default_argument_parser
from overdrive.engine import default_setup
from overdrive.engine import launch
from overdrive.evaluation import COCOEvaluator
from overdrive.evaluation import PascalVOCDetectionEvaluator


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create a COCO or Pascal VOC detection evaluator for the dataset.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.train.output_dir, "inference")

    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "coco":
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    raise NotImplementedError(f"no detection evaluator for the dataset {dataset_name} with the type {evaluator_type}")


class Trainer(DefaultTrainer):
    """
    Extend ``overdrive.engine.Trainer`` with dataset-specific evaluation.

    Evaluators are selected using each dataset's ``evaluator_type`` metadata.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
