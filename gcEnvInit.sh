

source /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.1.0+1205-58b501c780/poplar-ubuntu_20_04-3.1.0+6824-9c103dc348/enable.sh
source /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.1.0+1205-58b501c780/popart-ubuntu_20_04-3.1.0+6824-9c103dc348/enable.sh
mkdir -p /localdata/$USER/tmp
export TF_POPLAR_FLAGS=--executable_cache_path=/localdata/$USER/tmp
export POPTORCH_CACHE_DIR=/localdata/$USER/tmp
source poptorch_venv/bin/activate
# export POPLAR_LOG_LEVEL=INFO
# export POPLIBS_LOG_LEVEL=INFO
