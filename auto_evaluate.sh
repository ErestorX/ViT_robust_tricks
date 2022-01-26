#!/bin/bash

#VIT_VAL="doexp5 donegexp025l dosq4015"
VIT_VAL="doexp5 donegexp025l"
T2T_VAL="t p t_doexp05l t_donegexp05l t_donegexp025l"
NUM_PROC=$1
shift

#for v in {0..6}; do
echo "Evaluating the ViTs"
for v in {0,1}; do
  for ckpt in $VIT_VAL; do
    python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC Eval_robustness.py -version "$v" -ckpt "$ckpt" "$@"
  done
  python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC Eval_robustness.py -version "$v" "$@"
  python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC Eval_robustness.py -version "$v" -p "$@"
done
echo "Evaluating the T2Ts"
for ckpt in $T2T_VAL; do
  python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC Eval_robustness.py -version 2 -ckpt "$ckpt" "$@"
done
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC Eval_robustness.py -version 2 "$@"
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC Eval_robustness.py -version 2 -p "$@"