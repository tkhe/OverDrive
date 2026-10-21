import logging
from collections.abc import Mapping

__all__ = ["print_csv_format"]


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    if not isinstance(results, Mapping):
        raise TypeError(f"Expected results to be a Mapping, but got {type(results)}")

    if len(results) == 0:
        raise ValueError("Expected results to contain at least one task result")

    logger = logging.getLogger(__name__)
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info(f"copypaste: Task: {task}")
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join([f"{k[1]:.4f}" for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
