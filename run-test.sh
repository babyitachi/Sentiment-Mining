#!/bin/bash 

args=("$@")

if [ $# -ne 3 ]
    then echo "Please provie both inputext.txt and outputtext.txt files"
    else python run-test.py ${args[0]} ${args[1]} ${args[2]}
fi

sleep 5s