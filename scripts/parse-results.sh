#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR=$1

TOTAL_INJS=$(find $DIR -name 'stdout.txt' -type f 2>/dev/null | wc -l)
I=0

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

    I=$((I+1))
     PROGRESS=$(bc -l <<< "scale=2; $I * 100 / $TOTAL_INJS")
    echo -ne "Parsing $I/$TOTAL_INJS ($PROGRESS %)\r"
done

echo ""
echo "TRUE POSITIVE:  $TP"
echo "FALSE POSITIVE: $FP"
echo "FALSE NEGATIVE: $FN"
echo "TRUE NEGATIVE:  $TN"
