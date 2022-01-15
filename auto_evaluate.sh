#!/bin/bash

StringVal="doexp5 dosq4015 t p t_doexp05l t_donegexp05l"
#StringVal="doexp5 t p t_doexp05l"
NUM_PROC=$1
shift

#for v in {0..6}; do
for v in {0,1,2}; do
  for ckpt in $StringVal; do
    python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC Eval_robustness.py -version "$v" -ckpt "$ckpt" "$@"
  done
  python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC Eval_robustness.py -version "$v" "$@"
  python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC Eval_robustness.py -version "$v" -p "$@"
done