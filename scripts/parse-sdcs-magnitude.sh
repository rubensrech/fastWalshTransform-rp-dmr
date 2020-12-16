#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'Usage: $0 <campaign-dir>'
    exit -1
fi

CAMPAIGN_DIR=$1
OUT_FILE="sdcs-magnitude.txt"
NUM_REGEX='[0-9]+\.[0-9]+e[-+]?[0-9]+'

cd $CAMPAIGN_DIR

# Clean output file
echo "" > $OUT_FILE

echo "==========================================================================" >> $OUT_FILE
echo "=== Magnitude of the SDCs (absolute errors) ==============================" >> $OUT_FILE
echo "==========================================================================" >> $OUT_FILE

for class in 'TRUE_POSITIVE' 'FALSE_NEGATIVE'
do
    echo "> $class" >> $OUT_FILE

    for file in ./$class/*/out-vs-gold-stats.txt
    do
        if [[ -f "$file" ]];
        then
            # Extract values from stats file
            maxAbsErr=$(grep -oE "Max absolute err: $NUM_REGEX" $file | grep -oE $NUM_REGEX)
            minAbsErr=$(grep -oE "Min absolute err: $NUM_REGEX" $file | grep -oE $NUM_REGEX)
            avgAbsErr=$(grep -oE "Avg absolute err: $NUM_REGEX" $file | grep -oE $NUM_REGEX)
            echo "    * MAX: $maxAbsErr; MIN: $minAbsErr; AVG: $avgAbsErr" >> $OUT_FILE
        fi
    done
done

cat $OUT_FILE