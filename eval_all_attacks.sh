#!/bin/bash
NUM_PROC=$1
DATA=$2
CKA=$3

echo "Evaluating FGSM 0.031"
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 1 -epsilon 0.031 -$CKA -b 32
echo "Evaluating FGSM 0.062"
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 1 -epsilon 0.062 -$CKA -b 32

echo "Evaluating PGD 0.001"
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.001 -$CKA -b 32
echo "Evaluating PGD 0.003"
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.003 -$CKA -b 32
echo "Evaluating PGD 0.005"
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.005 -$CKA -b 32
echo "Evaluating PGD 0.01"
./auto_evaluate.sh $NUM_PROC -data $DATA -steps 40 -epsilon 0.01 -$CKA -b 32

