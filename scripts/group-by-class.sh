#!/bin/bash

CAMPAIGN_DIR=$1

cd $CAMPAIGN_DIR

for class in 'TRUE POSITIVE' 'TRUE NEGATIVE' 'FALSE POSITIVE' 'FALSE NEGATIVE'
do
    # Replace space with underscore
    DIR=$(tr -s ' ' '_' <<< "$class")
    # Create folder for each class
    [ ! -d "$DIR" ] && mkdir "$DIR"
    # Move fault instances to the folder according its classification
    grep -l "$class" ./*/stdout.txt | xargs -r -L 1 dirname | xargs -I{} mv {} $DIR
    # Count fault for each class
    faultsInClass=$(find "./$DIR/" -type d -name "fastWalshTransform*" | wc -l)
    echo "$class: $faultsInClass"
done