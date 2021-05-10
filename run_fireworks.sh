#Executable
#! /usr/bin/bash

# First deploy the dask cluster

python start_scheduler.py

# Wait a bit

sleep 10


# Now run the firetasks

python run_firetask.py








