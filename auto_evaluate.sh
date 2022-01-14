#!/bin/bash

StringVal="doexp5 dosq4015 t p t_doexp05l"
#StringVal="doexp5 t p t_doexp05l"

ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -data)
      ARGS="${ARGS} -data $2"
      shift # past argument
      shift # past value
      ;;
    -b)
      ARGS="${ARGS} -b $2"
      shift # past argument
      shift # past value
      ;;
    -epsilon)
      ARGS="${ARGS} -epsilon $2"
      shift # past argument
      shift # past value
      ;;
    -steps)
      ARGS="${ARGS} -steps $2"
      shift # past argument
      shift # past value
      ;;
    -step_size)
      ARGS="${ARGS} -step_size $2"
      shift # past argument
      shift # past value
      ;;
    -GPU)
      ARGS="${ARGS} -gpu $2"
      shift # past argument
      shift # past value
      ;;
    -CKA)
      ARGS="${ARGS} -CKA"
      shift # past argument
      ;;
    -CKA_single)
      ARGS="${ARGS} -CKA_single"
      shift # past argument
      ;;
    -all_exp)
      ARGS="${ARGS} -all_exp"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

#for v in {0..6}; do
for v in {0,1,2}; do
  for ckpt in $StringVal; do
    python3 Eval_robustness.py -version "$v" -ckpt "$ckpt" $ARGS
  done
  python3 Eval_robustness.py -version "$v" $ARGS
  python3 Eval_robustness.py -version "$v" -p $ARGS
done