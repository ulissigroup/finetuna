#! /usr/bin/bash

# First deploy the dask cluster in the background

set -m

./run_scheduler.sh & # start the primary process and put it in the background

# Start the second process

./run_fw.sh
















