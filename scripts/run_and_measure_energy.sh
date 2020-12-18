#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <num-iterations>"
    exit 1
fi

NUM_ITERATIONS=$1

cd ../

DATE=$(TZ="Europe/Berlin" date +"%d.%m.%Y-%H.%M")
OUTPUT_FILE="results/$NUM_ITERATIONS--$DATE.txt"

for ((i=1;i<=NUM_ITERATIONS;i++)); do

    # Execute in background
    ./fastWalshTransform -input inputs/input-bit-21.data -measureTime 1 >> $OUTPUT_FILE 2>> /dev/null &
    PID=$!

    # Calculate energy consumption
    ENERGY=0
    while ps -p $PID &>/dev/null; do
        CURR_POWER=$(cat /sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power0_input)
        ENERGY=$(bc <<< "$ENERGY + $CURR_POWER * 0.1")
        sleep 0.1
    done

    echo "> ITERATION $i => Total energy: $ENERGY mJ" >> $OUTPUT_FILE
    echo -e "\n\n" >> $OUTPUT_FILE

    echo "> ITERATION $i => Total energy: $ENERGY mJ"

done

