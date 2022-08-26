"""Run script

Can be used to run the steps of the pipeline by just providing a configuration
file and calling it with the flags for the respective steps that should be run.

For more information on the available flags and on how to call this script:
python run.py --help
"""
import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time

import attr
import toml

from funlib.run import run
from linajea.config import (load_config,
                            TrackingConfig)

logger = logging.getLogger(__name__)


def backup_and_copy_file(source, target, fn):
    target_fn = os.path.join(target, fn)
    if os.path.exists(target_fn):
        os.makedirs(os.path.join(target, "backup"), exist_ok=True)
        shutil.copy2(target_fn,
                     os.path.join(target, "backup",
                                  fn + "_backup" + str(int(time.time()))))
    if source is not None:
        source_fn = os.path.join(source, fn)
        if source_fn != target_fn:
            shutil.copy2(source_fn, target_fn)


def do_train(args, config, cmd):
    queue = config.train.job.queue
    num_gpus = 1
    num_cpus = config.train.job.num_workers
    flags = (['-P', config.train.job.lab]
             if config.train.job.lab is not None else [])
    run_cmd(args, config, cmd, "train",
            queue, num_gpus, num_cpus,
            flags=flags)


def do_predict(args, config, cmd):
    queue = 'interactive' if args.interactive else 'local'
    num_gpus = 0
    num_cpus = config.predict.job.num_workers
    flags = (['-P', config.predict.job.lab]
             if config.predict.job.lab is not None else [])
    run_cmd(args, config, cmd, "predict",
            queue, num_gpus, num_cpus,
            flags=flags)


def do_extract_edges(args, config, cmd):
    queue = 'interactive' if args.interactive else config.extract.job.queue
    num_gpus = 0
    num_cpus = config.extract.job.num_workers
    flags = (['-P', config.extract.job.lab]
             if config.extract.job.lab is not None else [])
    run_cmd(args, config, cmd, "extract",
            queue, num_gpus, num_cpus,
            flags=flags)


def do_solve(args, config, cmd, wait=True):
    queue = 'interactive' if args.interactive else config.solve.job.queue
    num_gpus = 0
    num_cpus = config.solve.job.num_workers
    flags = (['-P ' + config.solve.job.lab]
             if config.solve.job.lab is not None else [])
    flags.append('-W 60')

    if args.val_param_id is not None:
        cmd += ["--val_param_id", str(args.val_param_id)]
    if args.array_job:
        array_limit = 100
        if args.param_ids is not None:
            array_start = args.param_ids[0]
            array_end = args.param_ids[1]
            cmd += ["--param_id", '"$LSB_JOBINDEX"']
        else:
            array_start = 1
            array_end = len(config.solve.parameters)
            cmd += ["--param_list_idx", '"$LSB_JOBINDEX"']
    else:
        array_limit = None
        array_start = None
        array_end = None

        if args.param_ids is not None:
            cmd += ["--param_ids",
                    str(args.param_ids[0]),
                    str(args.param_ids[1])]
        elif args.param_id is not None:
            cmd += ["--param_id", str(args.param_id)]

    # if args.run_evaluate and (args.array_job or args.eval_array_job):
    if args.run_evaluate:
        wait = False

    jobid = run_cmd(args, config, cmd, "solve",
                    queue, num_gpus, num_cpus,
                    array_limit=array_limit,
                    array_start=array_start, array_end=array_end,
                    flags=flags, wait=wait)
    return jobid


def do_evaluate(args, config, cmd, jobid=None, wait=True):
    queue = 'interactive' if args.interactive else config.evaluate.job.queue
    num_gpus = 0
    num_cpus = 1
    flags = (['-P', config.evaluate.job.lab]
             if config.evaluate.job.lab is not None else [])
    flags.append('-W 30')

    if args.param_id is not None:
        cmd += ["--param_id", str(args.param_id)]
    if args.val_param_id is not None:
        cmd += ["--val_param_id", str(args.val_param_id)]

    if jobid is not None:
        if args.array_job:
            flags.extend(['-w', 'done({}[*])'.format(jobid)])
        else:
            flags.extend(['-w', 'done({})'.format(jobid)])

    if args.array_job or args.eval_array_job:
        array_limit = 100
        if args.param_ids is not None:
            array_start = args.param_ids[0]
            array_end = args.param_ids[1]
            cmd += ["--param_id", '"$LSB_JOBINDEX"']
        else:
            array_start = 1
            array_end = len(config.solve.parameters)
            cmd += ["--param_list_idx", '"$LSB_JOBINDEX"']
    else:
        array_limit = None,
        array_start = None
        array_end = None
    run_cmd(args, config, cmd, "evaluate",
            queue, num_gpus, num_cpus,
            array_limit=array_limit,
            array_start=array_start, array_end=array_end,
            flags=flags, wait=wait)


def run_cmd(args, config, cmd, job_name,
            queue, num_gpus, num_cpus,
            array_limit=0, array_size=0,
            array_start=None, array_end=None, flags=[], wait=True):
    if not args.local and not args.slurm and not args.gridengine:
        cmd = run(
            command=cmd,
            queue=queue,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            singularity_image=config.general.singularity_image,
            batch=not args.interactive,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            job_name=job_name,
            array_size=array_size,
            array_limit=array_limit,
            array_start=array_start,
            array_end=array_end,
            flags=flags)
    print(cmd)
    print(' '.join(cmd))

    if not args.array_job and not args.eval_array_job and \
       (args.slurm or args.gridengine):
        if args.slurm:
            if num_gpus > 0:
                cmd = ['sbatch', '../run_slurm_gpu.sh'] + cmd[1:]
            else:
                cmd = ['sbatch', '../run_slurm_cpu.sh'] + cmd[1:]
        elif args.gridengine:
            if num_gpus > 0:
                cmd = ['qsub', '../run_gridengine_gpu.sh'] + cmd[1:]
            else:
                cmd = ['qsub', '../run_gridengine_cpu.sh'] + cmd[1:]
        print(cmd)
        output = subprocess.run(cmd, check=True)
    else:
        if args.local:
            output = subprocess.run(
                cmd,
                check=True,
                encoding='UTF-8')
        else:
            output = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='UTF-8')

        jobid = None
        if not args.local:
            bsub_stdout_regex = re.compile(r"Job <(\d+)> is submitted*")
            logger.debug("Command output: %s" % output)
            print(output.stdout)
            print(output.stderr)
            match = bsub_stdout_regex.match(output.stdout)
            jobid = match.group(1)
            print(jobid)
            if wait and \
               not subprocess.run(["bwait", "-w", 'ended({})'.format(jobid)]):
                print("{} failed".format(cmd))
                exit()

        return jobid


def warn_if_not_abs_paths(config):
    rel_paths = []
    rel_path_keys = []
    if not os.path.isabs(config["general"]["setup_dir"]):
        rel_paths.append(config["general"]["setup_dir"])
        rel_path_keys.append("config.general.setup_dir")
    for ds in config["train_data"]["data_sources"]:
        if ds.get("tracksfile") and not os.path.isabs(ds["tracksfile"]):
            rel_paths.append(ds["tracksfile"])
            rel_path_keys.append("config.train_data.data_sources.tracksfile")
        if ds["datafile"]["filename"] is not None and \
           not os.path.isabs(ds["datafile"]["filename"]):
            rel_paths.append(ds["datafile"]["filename"])
            rel_path_keys.append(
                "config.train_data.data_sources.datafile.filename")
    for ds in config["validate_data"]["data_sources"]:
        if ds.get("tracksfile") is not None and not os.path.isabs(ds["tracksfile"]):
            rel_paths.append(ds["tracksfile"])
            rel_path_keys.append(
                "config.validate_data.data_sources.tracksfile")
        if ds["datafile"]["filename"] is not None and \
           not os.path.isabs(ds["datafile"]["filename"]):
            rel_paths.append(ds["datafile"]["filename"])
            rel_path_keys.append(
                "config.validate_data.data_sources.datafile.filename")
    for ds in config["test_data"]["data_sources"]:
        if ds.get("tracksfile") is not None and not os.path.isabs(ds["tracksfile"]):
            rel_paths.append(ds["tracksfile"])
            rel_path_keys.append("config.test_data.data_sources.tracksfile")
        if ds["datafile"]["filename"] is not None and \
           not os.path.isabs(ds["datafile"]["filename"]):
            rel_paths.append(ds["datafile"]["filename"])
            rel_path_keys.append(
                "config.test_data.data_sources.datafile.filename")

    if rel_paths:
        logger.warning(
            "If using run.py we recommend setting all paths in the config to"
            f"their absolute pathnames, {rel_path_keys} is/are relative "
            f"({rel_paths}). You may proceed, however there might be errors "
            "if not called from the respective setup dir")
        return False
    return True


def main():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='path to config file')
    parser.add_argument(
        '--checkpoint',
        type=int,
        default=-1,
        help='index of model checkpoint that should be used for prediction')
    parser.add_argument(
        '--train',
        action="store_true",
        dest='run_train',
        help='set if training step should be executed')
    parser.add_argument(
        '--predict',
        action="store_true",
        dest='run_predict',
        help='set if prediction step should be executed')
    parser.add_argument(
        '--extract_edges',
        action="store_true",
        dest='run_extract_edges',
        help='set if extract edges step should be executed')
    parser.add_argument(
        '--solve',
        action="store_true",
        dest='run_solve',
        help='set if solving (tracking) step should be executed')
    parser.add_argument(
        '--evaluate',
        action="store_true",
        dest='run_evaluate',
        help='set if evaluation step shuld be executed')
    parser.add_argument(
        '--validation',
        action="store_true",
        help=('set if validation data should be sued (instead of test data),'
              'applies to postprocessing steps but not to training step'))
    parser.add_argument(
        '--validate_on_train',
        action="store_true",
        help=('can be used to perform validation on training data'
              'mostly for debugging purposes'))
    parser.add_argument(
        '--param_id',
        type=int,
        default=None,
        help=('set to the param_id value of a specific set of weights,'
              'if only this set should be evaluated (has to exist in the '
              'database)'))
    parser.add_argument(
        '--val_param_id',
        type=int,
        default=None,
        help=('if you want to solve on the test data using a specific set of '
              'weights that has been evaluated on the validation data, set '
              'this to the respective param_id value of that set in the'
              'validation database'))
    parser.add_argument(
        '--param_ids',
        default=None,
        nargs='+',
        help=('defines a number of set of weights that should be processed. '
              'if two values are supplied it is interpreted as a range and '
              'and all sets of weights with a param_id value in that range '
              'will be processed. otherwise it is interpreted as a list of '
              'param_id values, and the respective sets of weights in the '
              'database will be processed. irrespectively all values have '
              'to exist in the database'))
    parser.add_argument(
        "--local",
        action="store_true",
        help=('set if steps should be run on the local machine and not on '
              'separate jobs on a HPC cluster'))
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=('set if steps should be run on a slurm-based cluster '
              '(experimental)'))
    parser.add_argument(
        "--gridengine",
        action="store_true",
        help=('set if steps should be run on a slurm-based cluster '
              '(experimental)'))
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=('if running on an lsf cluster, set if it should be run on an '
              'interactive node'))
    parser.add_argument(
        '--array_job',
        action="store_true",
        help=('if running on an lsf cluster, set if each set of weights for '
              'the solving and evaluation steps should be run as an '
              'independant job'))
    parser.add_argument(
        '--eval_array_job',
        action="store_true",
        help=('similar to array_job, only applies to evaluation step, whereas'
              'array_job applies to both, solving and evaluation'))

    parser.add_argument(
        '--wait_job_id',
        type=str,
        default=None,
        help=('if runnning on an lsf cluster, wait for the job with this id '
              'before starting'))
    parser.add_argument(
        "--no_block_after_eval",
        dest="block_after_eval",
        action="store_false",
        help=('if running on an lsf cluster, set this to immediatly return '
              'to the shell instead of waiting (blocking) until an evaluation '
              'job is finished, job will be executed in the background'))

    args = parser.parse_args()
    config = load_config(args.config)
    config["path"] = args.config
    is_abs = warn_if_not_abs_paths(config)
    config = TrackingConfig(**config)

    setup_dir = config.general.setup_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    is_new_run = not os.path.exists(setup_dir)

    if is_abs:
        os.makedirs(setup_dir, exist_ok=True)

        if not is_new_run:
            config_dir = os.path.dirname(os.path.abspath(args.config))
            if config_dir != os.path.abspath(setup_dir) and \
               "tmp_configs" not in args.config:
                raise RuntimeError(
                    "overwriting config with external config file (%s - %s)",
                    args.config, setup_dir)
        if "tmp_configs" not in args.config:
            backup_and_copy_file(os.path.dirname(args.config),
                                 setup_dir,
                             os.path.basename(args.config))
        if is_new_run:
            config.path = os.path.join(setup_dir, os.path.basename(args.config))

        os.chdir(setup_dir)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tmp_configs", exist_ok=True)

    logging.basicConfig(
        level=config.general.logging,
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    os.environ["GRB_LICENSE_FILE"] = "/misc/local/gurobi-9.1.2/gurobi.lic"
    run_steps = []
    if args.run_train:
        run_steps.append("01_train.py")
    if args.run_predict:
        run_steps.append("02_predict_blockwise.py")
    if args.run_extract_edges:
        run_steps.append("03_extract_edges.py")
    if args.run_solve:
        run_steps.append("04_solve.py")
    if args.run_evaluate:
        run_steps.append("05_evaluate.py")

    config.path = os.path.join("tmp_configs", "config_{}.toml".format(
        time.time()))
    config_dict = attr.asdict(config)
    config_dict['solve']['grid_search'] = False
    config_dict['solve']['random_search'] = False
    with open(config.path, 'w') as f:
        toml.dump(config_dict, f)

    jobid = args.wait_job_id
    for step in run_steps:
        cmd = ["python",
               os.path.join(script_dir, step)]
        if args.checkpoint > 0:
            cmd.append("--checkpoint")
            cmd.append(str(args.checkpoint))
        if args.validation:
            cmd.append("--validation")
        if args.validate_on_train:
            cmd.append("--validate_on_train")

        cmd += ["--config", config.path]

        if step == "01_train.py":
            do_train(args, config, cmd)
        elif step == "02_predict_blockwise.py":
            do_predict(args, config, cmd)
        elif step == "03_extract_edges.py":
            do_extract_edges(args, config, cmd)
        elif step == "04_solve.py":
            jobid = do_solve(args, config, cmd)
        elif step == "05_evaluate.py":
            do_evaluate(args, config, cmd, jobid=jobid,
                        wait=args.block_after_eval)

        else:
            raise RuntimeError("invalid processing step! %s", step)


if __name__ == "__main__":
    main()
