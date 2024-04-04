#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
python -m torch.distributed.launch --nproc_per_node=${NGPUS} train_track.py --launcher pytorch ${PY_ARGS}
#python -m torch.distributed.launch --nproc_per_node=4 train_track.py --launcher pytorch --cfg_file cfgs/nus_models/nuscenes_bus_qapillar.yaml --extra_tag bus --fix_random_seed

