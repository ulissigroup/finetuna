# start_scheduler.py
from distributed import Client, LocalCluster
from tornado.ioloop import IOLoop
from utilities import extract_job_parameters
import os

if __name__ == '__main__':

    # Get the number of workers from the job_params.yml file
#    workers = extract_job_parameters(int(os.environ['JOB_ID']))['cores']
    workers=20
    loop = IOLoop()
    cluster = LocalCluster(n_workers=workers, threads_per_worker=1, memory_limit=0)
    client = Client(cluster)
    client.write_scheduler_file('my-scheduler.json')
    loop.start()  # keeps the scheduler running
    loop.close()
