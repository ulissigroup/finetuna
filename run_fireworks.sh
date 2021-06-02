#! /usr/bin/bash

# First deploy the dask cluster in the background

#set -m

./run_scheduler.sh & # start the primary process and put it in the background

# Use 20 cores to run VASP binary
export VASP_COMMAND="mpirun -np 20 /opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std"
# Start the second process
cd .. # launch in PVC /home/jovyan
rlaunch rapidfire --nlaunches infinite --sleep 10
#./add_fw.sh
















