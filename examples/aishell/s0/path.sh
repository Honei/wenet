export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/server/x86/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

#export LD_LIBRARY_PATH=/home/users/xiongxinlei/opt/nccl/nccl-2.4.2/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/users/xiongxinlei/opt/cudnn/cuda11.1/cudnn-8.1.1/lib64/:$LD_LIBRARY_PATH
