#!/bin/bash

# Auxiliary file to launch experiments

sync ./ # Force sync for mounted filesystems

# Export the cuda lib path, if existing.
echo `env | grep LD`
if [ -e /usr/local/cuda-9.0 ]; then
  export PATH=$PATH:/usr/local/cuda-9.0/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/extras/CUPTI/lib64
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
  export CUDA_ROOT=/usr/local/cuda-9.0
fi
echo `env | grep LD`


# Change the base directory if it was not launched from here.
BASEDIR=$(dirname "$0")
if [ $BASEDIR != `pwd` ];
then
  pushd $BASEDIR
fi

# Get the current git version
GIT_VERSION=`git rev-parse --short HEAD`

# Set which GPU to use
export CUDA_VISIBLE_DEVICES="2"

# Experiment options
exp="mpii"
mode="train"
cate="mpii_singleperson"
target="trial-00"

OUTDIR="output"
mkdir -p ${OUTDIR}
logid="${cate}_${target}_${GIT_VERSION}"
pyexec='python3'

# Build the command to be launched.
if [ "" == "$1" ]; then
  cmd="exp/${exp}/${mode}_${cate}.py ${OUTDIR}/$logid"
elif [ "-I" == "$1" ]; then
  cmd=''
else
  cmd="$1 ${OUTDIR}/$logid"
fi

# Launch it!
eval ${pyexec} ${cmd}
