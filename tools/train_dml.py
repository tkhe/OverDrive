from overdrive.checkpoint import Checkpointer
from overdrive.engine import DMLTrainer
from overdrive.engine import default_argument_parser
from overdrive.engine import launch

from train_net import build_evaluator
from train_net import setup


class Trainer(DMLTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        checkpointer = Checkpointer(model, cfg.train.output_dir)
        Trainer.load_initial_weights(cfg, checkpointer, resume=args.resume)
        return Trainer.test(cfg, model)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
