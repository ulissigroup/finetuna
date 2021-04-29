# start_scheduler.py
from distributed import Client, LocalCluster
from tornado.ioloop import IOLoop

if __name__ == '__main__':
    loop = IOLoop()
    cluster = LocalCluster(n_workers=20, threads_per_worker=1, memory_limit=0)
    client = Client(cluster)
    client.write_scheduler_file('my-scheduler.json')
    loop.start()  # keeps the scheduler running
    loop.close()
