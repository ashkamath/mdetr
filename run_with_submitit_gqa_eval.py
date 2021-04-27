# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import copy
import itertools
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Dict

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import submitit

import scripts.eval_gqa as detection


def parse_args():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser("Submitit detection", parents=[detection_parser])

    parser.add_argument("--partition", default=None, type=str, help="Partition where to submit")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=4, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4300, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--mail", default="", type=str, help="Email this user when the job finishes if specified")
    return parser.parse_args()


def get_shared_folder(args) -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    init_file = get_shared_folder(args) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def grid_parameters(grid: Dict):
    """
    Yield all combinations of parameters in the grid (as a dict)
    """
    grid_copy = dict(grid)
    # Turn single value in an Iterable
    for k in grid_copy:
        if not isinstance(grid_copy[k], Iterable):
            grid_copy[k] = [grid_copy[k]]
    for p in itertools.product(*grid_copy.values()):
        yield dict(zip(grid.keys(), p))


def sweep(executor: submitit.Executor, args: argparse.ArgumentParser, hyper_parameters: Iterable):
    jobs = []
    with executor.batch():
        for grid_data in hyper_parameters:
            tmp_args = copy.deepcopy(args)
            tmp_args.dist_url = get_init_file(args).as_uri()
            tmp_args.output_dir = args.job_dir
            for k, v in grid_data.items():
                assert hasattr(tmp_args, k)
                setattr(tmp_args, k, v)
            trainer = Trainer(tmp_args)
            job = executor.submit(trainer)
            jobs.append(job)
    print("Sweep job ids:", [job.job_id for job in jobs])


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        socket_name = os.popen("ip r | grep default | awk '{print $5}'").read().strip("\n")
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        os.environ["MDETR_CPU_REDUCE"] = "1"

        import scripts.eval_gqa as detection

        self._setup_gpu_args()
        detection.main(self.args)

    def checkpoint(self):
        import os
        from pathlib import Path

        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(args) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    partition = args.partition
    timeout_min = args.timeout
    kwargs = {}
    if partition is not None:
        kwargs["slurm_partition"] = partition

    executor.update_parameters(
        mem_gb=45 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(name="detectransformer")
    if args.mail:
        executor.update_parameters(additional_parameters={"mail-user": args.mail, "mail-type": "END"})

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
