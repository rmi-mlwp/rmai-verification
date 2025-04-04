import argparse
import logging
from utils.files import yaml_file_type

# Define log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"



LOG = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Run a verification pipeline based on a config-file"
)

parser.add_argument(
    "-c,","--config",
    required=True,
    type=yaml_file_type,
    help="Path to the YAML configuration file"
)

parser.add_argument(
    "--n_workers",
    default=4,
    type=int,
    help="Number of dask workers"
)

parser.add_argument(
    "--threads_per_worker",
    default=1,
    type=int,
    help="Number of threads per dask worker"
)

args = parser.parse_args()

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from verification.verification import Verification

if __name__ == "__main__":

    # cluster = LocalCluster(
    #     n_workers=args.n_workers,
    #     threads_per_worker=args.threads_per_worker,
    #     processes=True,
    # )

    cluster = SLURMCluster(
        queue = "standard",
        account = "project_465000527",
        cores=64,
        processes=4,
        memory="240GB",
        interface="hsn0",
    )

    cluster.scale(10)
    client = Client(cluster)


    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format=LOG_FORMAT, 
        datefmt=DATE_FORMAT,
        handlers=[
            #logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()          # Log to console
        ]
    )

    verif = Verification(args.config)
    try:
        verif.verify()
    except:
        LOG.error("Error during verfication closing down dask cluster",exc_info=True)
        client.close()
        cluster.close()
    
    client.close()
    cluster.close()




