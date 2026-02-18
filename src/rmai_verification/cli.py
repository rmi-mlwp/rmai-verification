import argparse
import sys
import logging
import time

# Define log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG = logging.getLogger(__name__)


def run_local_gpu(args):
    # GPU mode: NO dask.distributed, NO LocalCluster
    import dask
    from .verification.verification import Verification
    from .utils.profiling import Profiler, set_profiler
    
    print("Running in GPU mode: using dask threads scheduler (single process, single GPU)...")

    # Force threaded scheduler (single process, no serialization)
    dask.config.set(scheduler="threads")

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler()],
    )

    LOG.info("Running in GPU mode: dask threads scheduler (single process, single GPU)")

    # Initialize profiler
    profiler = Profiler(mode="local_gpu")
    set_profiler(profiler)
    profiler.start()

    verif = Verification(args.CONFIG)
    try:
        verif.verify()
    except Exception:
        LOG.error("Error during verification", exc_info=True)
        sys.exit(1)
    finally:
        profiler.end()
        profiler.print_summary()
        
        # Save profiling results if requested
        if args.profile_output:
            profiler.save_to_file(args.profile_output)


def run_local(args):
    # Only import the necessary modules if function is called
    # to avoid unnecessary slow imports at the top level
    from dask.distributed import Client, LocalCluster
    from .verification.verification import Verification
    from .utils.profiling import Profiler, set_profiler
    
    print(f"Starting local dask cluster with {args.n_workers} workers and {args.threads_per_worker} threads per worker...")
    
    # Initialize profiler
    profiler = Profiler(mode="local")
    set_profiler(profiler)
    profiler.start()
    
    # Time cluster startup
    cluster_start = time.time()
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
    )
    client = Client(cluster)
    cluster_startup_time = time.time() - cluster_start
    profiler.record_cluster_startup(cluster_startup_time)

    logging.basicConfig(
        level=logging.INFO,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format=LOG_FORMAT, 
        datefmt=DATE_FORMAT,
        handlers=[
            #logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()          # Log to console
        ]
    )

    verif = Verification(args.CONFIG)
    try:
        verif.verify()
    except:
        LOG.error("Error during verfication closing down dask cluster",exc_info=True)
        client.close()
        cluster.close()
        sys.exit(1)
    finally:
        profiler.end()
        profiler.print_summary()
        
        # Save profiling results if requested
        if args.profile_output:
            profiler.save_to_file(args.profile_output)
        
        client.close()
        cluster.close()

def run_slurm(args):
    # Only import the necessary modules if function is called
    # to avoid unnecessary slow imports at the top level
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster
    from .verification.verification import Verification
    from .utils.profiling import Profiler, set_profiler, wait_for_workers
    
    print(f"Starting SLURM cluster with {args.jobs} jobs, {args.cores} cores per job, {args.memory} memory per job, "
          f"and walltime {args.walltime}...")
    
    # Initialize profiler
    profiler = Profiler(mode="slurm")
    set_profiler(profiler)
    profiler.start()
    
    # Time cluster startup
    cluster_start = time.time()
    cluster = SLURMCluster(
        queue = args.queue,
        account = args.account,
        cores = args.cores,
        processes = args.processes,
        memory = args.memory,
        interface = args.interface,
        walltime = args.walltime,
        job_extra_directives = args.job_extra_directives,
    )
    cluster.scale(jobs=args.jobs)
    client = Client(cluster)
    cluster_startup_time = time.time() - cluster_start
    profiler.record_cluster_startup(cluster_startup_time)

    # Wait for workers and measure queuing time
    expected_workers = args.jobs * args.processes
    queuing_time = wait_for_workers(client, expected_workers, timeout=args.worker_timeout)
    profiler.record_slurm_queuing(queuing_time)

    logging.basicConfig(
        level=logging.INFO,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format=LOG_FORMAT, 
        datefmt=DATE_FORMAT,
        handlers=[
            #logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()          # Log to console
        ]
    )

    verif = Verification(args.CONFIG)
    try:
        verif.verify()
    except:
        LOG.error("Error during verfication closing down dask cluster",exc_info=True)
        client.close()
        cluster.close()
        sys.exit(1)
    finally:
        profiler.end()
        profiler.print_summary()
        
        # Save profiling results if requested
        if args.profile_output:
            profiler.save_to_file(args.profile_output)
        
        client.close()
        cluster.close()

    verif = Verification(args.CONFIG)
    try:
        verif.verify()
    except:
        LOG.error("Error during verfication closing down dask cluster",exc_info=True)
        client.close()
        cluster.close()
        sys.exit(1)


def main():

    parser = argparse.ArgumentParser(description="RMAI Verification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    local_parser = subparsers.add_parser(
        "local", help="Run the verification pipeline based on a config-file on a local dask cluster"
    )

    local_parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Run in GPU mode (CuPy + dask threads scheduler, no distributed)"
    )

    local_parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="Number of dask workers"
    )

    local_parser.add_argument(
        "--threads_per_worker",
        default=1,
        type=int,
        help="Number of threads per dask worker"
    )
    
    local_parser.add_argument(
        "--profile_output",
        type=str,
        default=None,
        help="Path to save profiling results as JSON (optional)"
    )

    slurm_parser = subparsers.add_parser(
        "slurm", help="Run the verification pipeline based on a config-file on a slurm cluster"
    )

    slurm_parser.add_argument(
        "--queue",
        type=str,
        help="Destination queue for the worker jobs"
    )

    slurm_parser.add_argument(
        "--account",
        type=str,
        help="Account to charge the jobs to"
    )

    slurm_parser.add_argument(
        "--cores",
        type=int,
        default=8,
        help="Total number of CPU cores on which all worker threads inside a job will run"
    )

    slurm_parser.add_argument(
        "--memory",
        type=str,
        default="64GB",
        help="Total amount of memory to be used by all workers inside a job"
    )

    slurm_parser.add_argument(
        "--interface",
        type=str,
        default="hsn0",
        help="Network interface to use for the dask workers"
    )

    slurm_parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="How many jobs to scale with."
    )

    slurm_parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="How many processes to cut up the job into. (not used at the moment)"
    )

    slurm_parser.add_argument(
        "--walltime",
        type=str,
        default="01:00:00",
        help="Walltime for the jobs in the format HH:MM:SS"
    )

    slurm_parser.add_argument(
        "--job_extra_directive",
        dest="job_extra_directives",
        action="append",
        default=[],
        help="Extra SBATCH directives. Can be used multiple times, e.g. "
            "--job-extra-directive='--output=/path/%j.out' "
            "--job-extra-directive='--error=/path/%j.err'"
    )
    
    slurm_parser.add_argument(
        "--worker_timeout",
        type=int,
        default=300,
        help="Maximum time in seconds to wait for SLURM workers to become available"
    )
    
    slurm_parser.add_argument(
        "--profile_output",
        type=str,
        default=None,
        help="Path to save profiling results as JSON (optional)"
    )

    parser.add_argument(
        "CONFIG",
        type=str,
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()
    
    if args.use_gpu and (args.n_workers != 1 or args.threads_per_worker != 1):
        LOG.warning(
            "GPU mode ignores --n_workers and --threads_per_worker "
            "(threads scheduler, single process)"
        )

    if args.command == "local":
        if getattr(args, "use_gpu", False):
            run_local_gpu(args)
        else:
            run_local(args)
    elif args.command == "slurm":
        run_slurm(args)
    elif not args.command:
        parser.print_help()
        sys.exit(1)
    else:
        LOG.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format=LOG_FORMAT, 
        datefmt=DATE_FORMAT,
        handlers=[
            #logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()          # Log to console
        ]
    )

    LOG.info("Starting RMAI Verification CLI")
    main()

    

