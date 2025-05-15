#!/bin/bash

# Path to your base directory (adjust as needed)
BASE_DIR="$HOME/big_data/Stress/outputs"

# Loop through P001 to P059
for i in $(seq -w 1 59); do
    FOLDER="P0${i}"
    for j in $(seq -w 1 11); do
        VIDEO_FOLDER="${BASE_DIR}/${FOLDER}/tsk${j}_video"

        cd "$VIDEO_FOLDER"
        files=$(( $(ls -l | wc -l) - 2 ))
        frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 ../../../Recordings_30_fps/${FOLDER}/videos/tsk${j}_video.mp4)
        if [ "$files" -ne "$frames" ]; then
            echo "Directory $VIDEO_FOLDER has $files files but $frames frames."
        elif [ "$frames" -ne 3602 ]; then
            echo "-Directory $VIDEO_FOLDER has exactly $frames files and frames."
        fi
    done
done