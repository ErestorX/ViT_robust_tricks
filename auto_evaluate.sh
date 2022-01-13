#!/bin/bash

StringVal="doexp5 dosq4015 t p t_doexp05l"
#StringVal="doexp5 t p t_doexp05l"

for v in {0..6}; do
#for v in {3, 5}; do
  for ckpt in $StringVal; do
    python3 Eval_robustness.py --data "$1" --version "$v" --ckpt "$ckpt"
  done
  python3 Eval_robustness.py --data "$1" --version "$v"
  python3 Eval_robustness.py --data "$1" --version "$v" -p
done

python3 combine_eval_summaries.py