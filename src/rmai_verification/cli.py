import argparse
import sys
import logging

# Define log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG = logging.getLogger(__name__)

def run_local(args):
    # Only import the necessary modules if function is called
    # to avoid unnecessary slow imports at the top level
    from dask.distributed import Client, LocalCluster
    from .verification.verification import Verification
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
    )
    client = Client(cluster)

        
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

def run_slurm(args):
    # Only import the necessary modules if function is called
    # to avoid unnecessary slow imports at the top level
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster
    from .verification.verification import Verification
    import sys
    log_file = open("output.log", "a", buffering=1)  # buffering=1 for line buffering
    sys.stdout = log_file
    sys.stderr = log_file
    print("Starting SLURM cluster with the following parameters:")
    print(f"  Queue: {args.queue}")
    print(f"  Cores: {args.cores}")
    print(f"  Memory: {args.memory}")
    # print(f"  Interface: {args.interface}")
    # print(f"  Walltime: {args.time}")
    print(f"  Job extra directives: --qos=normal")
    print("Starting RMAI Verification CLI")
    cluster = SLURMCluster(
        queue = args.queue,
        account = args.account,
        cores = args.cores,
        #processes = args.processes,
        interface = "ib0",
        memory = args.memory,
        job_extra_directives = ["--qos=normal"],
        walltime = "02:00:00",
    )
    cluster.scale(jobs=3)
    client = Client(cluster)

    logging.basicConfig(
        level=logging.INFO,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format=LOG_FORMAT, 
        datefmt=DATE_FORMAT,
        handlers=[
            logging.FileHandler("output.log"),  # Log to a file
            # logging.StreamHandler()          # Log to console
        ]
    )
    print("Dask SLURM cluster started")
    print("Begin verification process")
    verif = Verification(args.CONFIG)
    print("Verification object created, starting verification")
    try:
        verif.verify()
        print("Verification process completed successfully")
    except:
        LOG.error("Error during verfication closing down dask cluster",exc_info=True)
        print("Error during verification, closing down dask cluster")
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

    # slurm_parser.add_argument(
    #     "--interface",
    #     type=str,
    #     default="hsn0",
    #     help="Network interface to use for the dask workers"
    # )
    parser.add_argument(
        "CONFIG",
        type=str,
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()

    if args.command == "local":
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
            logging.FileHandler("output.log", mode='w'),  # Log to a file
            # logging.StreamHandler()          # Log to console
        ]
    )

    LOG.info("Starting RMAI Verification CLI")
    main()

    

