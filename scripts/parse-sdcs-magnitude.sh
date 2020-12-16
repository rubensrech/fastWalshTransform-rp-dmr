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
NUM_REGEX='[0-9]+\.[0-9]+e[-+]?[0-9]+'


if ls ./$CAMPAIGN_DIR/fastWalshTransform* 1> /dev/null 2>&1;
then
    echo "You must run 'group-by-class.sh' before"
    exit -2
fi

cd $CAMPAIGN_DIR

# Clean output file
echo -n "" > $OUT_FILE

echo "==========================================================================" >> $OUT_FILE
echo "=== Magnitude of the SDCs ($ERR_METRIC errors) ==============================" >> $OUT_FILE
echo "==========================================================================" >> $OUT_FILE

for class in 'TRUE_POSITIVE' 'FALSE_NEGATIVE'
do
    echo "> $class" >> $OUT_FILE

    for file in ./$class/*/out-vs-gold-stats.txt
    do
        if [[ -f "$file" ]];
        then
            # Extract values from stats file
            maxErr=$(grep -oE "Max $ERR_METRIC err: $NUM_REGEX" $file | grep -oE $NUM_REGEX)
            minErr=$(grep -oE "Min $ERR_METRIC err: $NUM_REGEX" $file | grep -oE $NUM_REGEX)
            avgErr=$(grep -oE "Avg $ERR_METRIC err: $NUM_REGEX" $file | grep -oE $NUM_REGEX)
            echo "    * MAX: $maxErr; MIN: $minErr; AVG: $avgErr" >> $OUT_FILE
        fi
    done
done


echo "Results available in: $CAMPAIGN_DIR/$OUT_FILE"
