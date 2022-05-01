#!/bin/bash 

if [ -z $2 ]
    then echo "Please provide both training data directory and model save directory "
    else python3 run-train.py $0 $1
fi