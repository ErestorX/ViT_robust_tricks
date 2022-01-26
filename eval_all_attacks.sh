#!/bin/bash
NUM_PROC=$1
DATA=$2
CKA=$3

./auto_evaluate.sh $NUM_PROC -data $DATA -steps 1 -epsilon 0.031 -$CKA
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 1 -epsilon 0.062 -$CKA

./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.001 -$CKA
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.003 -$CKA
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.005 -$CKA
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.01 -$CKA

