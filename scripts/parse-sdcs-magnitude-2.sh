#!/bin/bash

if [[ $# -eq 0 ]] ; then
   echo 'Usage: $0 <campaign-dir> [<err-metric=absolute|relative*>]'
   exit -1
fi

ERR_METRIC='relative'

if [[ $# -eq 2 ]] ; then
   ERR_METRIC=$2
fi

CAMPAIGN_DIR=$1
OUT_FILE="sdcs-magnitude.txt"

FLT_REGEX='[0-9]+\.[0-9]+e[-+]?[0-9]+'
INT_REGEX='[0-9]+'

cd $CAMPAIGN_DIR

# Print header
echo "error_model,detection_outcome,avg_err,diff_vals_count" > $OUT_FILE

for errModel in 'RANDOM_VALUE' 'FLIP_SINGLE_BIT' 'FLIP_TWO_BITS'
do
    for class in 'TRUE_POSITIVE' 'FALSE_NEGATIVE'
    do
        for file in ./$errModel/$class/*/out-vs-gold-stats.txt
        do
            if [[ -f "$file" ]];
            then
                # Extract values from stats file
                avgErr=$(grep -oE "Avg $ERR_METRIC err: $FLT_REGEX" $file | grep -oE $FLT_REGEX)
                difValsCount=$(grep -oE "Number of diff values: $INT_REGEX" $file | grep -oE $INT_REGEX)
                
                echo "$errModel,$class,$avgErr,$difValsCount" >> $OUT_FILE
            fi
        done
    done
done

echo "Results available in: $CAMPAIGN_DIR/$OUT_FILE"
