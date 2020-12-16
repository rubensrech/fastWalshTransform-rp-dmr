#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR=$1

if ls ./$DIR/fastWalshTransform* 1> /dev/null 2>&1;
then

    TP=0
    FP=0
    FN=0
    TN=0

    for f in ./$DIR/fastWalshTransform*/stdout.txt
    do
        grep 'TRUE POSITIVE' $f > /dev/null; if [ $? -eq 0 ]; then TP=$((TP+1)); fi
        grep 'FALSE POSITIVE' $f > /dev/null; if [ $? -eq 0 ]; then FP=$((FP+1)); fi
        grep 'FALSE NEGATIVE' $f > /dev/null; if [ $? -eq 0 ]; then FN=$((FN+1)); fi
        grep 'TRUE NEGATIVE' $f > /dev/null; if [ $? -eq 0 ]; then TN=$((TN+1)); fi
    done

    echo "TRUE POSITIVE: $TP"
    echo "FALSE POSITIVE: $FP"
    echo "FALSE NEGATIVE: $FN"
    echo "TRUE NEGATIVE: $TN"

else

    for class in 'TRUE POSITIVE' 'FALSE POSITIVE' 'FALSE NEGATIVE' 'TRUE NEGATIVE'
    do
        classDir=$(tr -s ' ' '_' <<< "$class")
        count=$(find "./$DIR/$classDir/" -type d -name "fastWalshTransform*" | wc -l)
        echo "$class: $count"
    done

fi
