#!/bin/bash

# Path to your base directory (adjust as needed)
BASE_DIR="$HOME/big_data/Stress/outputs"

# Loop through P001 to P059
for i in $(seq -w 1 59); do
    FOLDER="P0${i}"
    for j in $(seq -w 1 12); do
        VIDEO_FOLDER="${BASE_DIR}/${FOLDER}/tsk${j}_video"

        find "$VIDEO_FOLDER" -mindepth 1 -maxdepth 1 -type d -not -name "*_000" -not -name "processed*" -print
    done
done