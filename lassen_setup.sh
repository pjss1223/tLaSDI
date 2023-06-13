#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 60
#BSUB -o setup.log

datadir="/p/gpfs1/${USER}/tLaSDI_data"

cp -r $datadir data
mkdir model
mkdir outputs

# Set up virtual environment
# Adapted from install-opence-1.8.0.sh
# https://lc.llnl.gov/confluence/display/LC/2023/01/11/Open-CE+1.8.0+for+Lassen

# NOTE: install open-ce in a directory with plenty of free space, like /usr/workspace/$USER
envname="opence-1.8.0"

# Load cuda/11.4.1
module load cuda/11.4.1

# install to anaconda subdirectory within current working directory
installdir="$(pwd)/anaconda"

# install conda
bash "/collab/usr/global/tools/opence/blueos_3_ppc64le_ib_p9/$envname/Miniconda3-py39_4.12.0-Linux-ppc64le.sh" -b -f -p "$installdir"

# activate conda environment
source "$installdir/bin/activate"

# create an opence environment in conda (Python-3.9)
conda update -n base -c defaults conda
conda create -y -n "$envname" python=3.9

# activate the opence environment
conda activate "$envname"

# register LLNL SSL certificates
conda config --env --set ssl_verify /etc/pki/tls/cert.pem

# register LC's local conda channel for Open-CE
condachannel="/collab/usr/global/tools/opence/${SYS_TYPE}/$envname/condabuild-py3.9-cuda11.4"
conda config --env --prepend channels "file://$condachannel"

# install tLaSDI packages
conda install -y pytorch=1.13.0=cuda11.4_py39_1
conda install -y scikit-learn=1.2.2
conda install -y matplotlib=3.7.1

echo "Created conda env:"
echo "  $envname"
echo
echo "To activate:"
echo "  source anaconda/bin/activate"
echo "  module load cuda/11.4.1"
echo
echo "To see available packages:"
echo "  conda search | grep condabuild-py3.9-cuda11.4"
echo
echo "To install packages:"
echo "  conda install -y pytorch=1.13.0=cuda11.4_py39_1"
echo "    Note: the above values can be found as the first 3 fields of the conda search command shown above"
echo
