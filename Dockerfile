# Use NVIDIA CUDA as base image and run the same installation as in the other packages.
# The version of cudatoolkit must match those of the base image, see Dockerfile.pytorch
FROM ulissigroup/vasp:atomate_stack_mlp

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]

USER $NB_UID

# Install Jupyter Notebook, Lab, and Hub
# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change

RUN conda install --quiet --yes \
    'notebook' \
    'ase' \
    'xlrd>=1.0.0' \
    'jupyterhub' \
    'matplotlib' \
    'plotly' \
    'tqdm' \
    'cmake' \
    'openblas' \
    'liblapacke' \
    'ipykernel' \
    'jupyterlab' &&\
    conda install --quiet --yes \
    'ipywidgets' -c conda-forge && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

EXPOSE 8888

# Install htop

RUN sudo apt update && sudo apt-get install htop

# Install flare_pp 
RUN BLA_VENDOR=OpenBLAS pip install flare_pp
ENV LD_LIBRARY_PATH "/opt/conda/lib:$LD_LIBRARY_PATH"

#RUN wget https://raw.githubusercontent.com/hackingmaterials/automatminer/master/requirements.txt
#RUN sed -i '/sci/d' ./infile
#RUN pip install git+https://github.com/hackingmaterials/automatminer.git

# Install Vasp Interactive in the image
RUN pip install git+https://github.com/ulissigroup/vasp-interactive.git && pip install deepdiff

# Getting issues running mo-wulff-worklow because PPs not compatible with pymatgen
RUN pmg config --add PMG_VASP_PSP_DIR $HOME/atomate/POTCARs/ && \
    pmg config --add PMG_DEFAULT_FUNCTIONAL PBE

# Switch back to jovyan to avoid accidental container runs as root
WORKDIR $HOME/
#COPY start_scheduler.py /home/jovyan/
# Launch the dask cluster in the container. This will also output the scheduler_file.json
#RUN python start_scheduler.py
#CMD ["python3", "start_scheduler.py"]
ENV PYTHONPATH "$PYTHONPATH/home/jovyan/al_mlp_repo:/home/jovyan/mo-wulff-workflow"
USER root
RUN fix-permissions /home/$NB_USER

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

ENV NB_PREFIX /
#CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
