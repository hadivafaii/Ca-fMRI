#!/bin/bash

n_clusters=${1:-5}
mode=${2:-"ca2"}
metric=${3:-"correlation"}
task=${4:-"rest"}
# save_masks=${5:-"$true"}

cd ..

run_hier () {
  # for n_rois in $(seq 100 100 900); do
  for i in $(seq 3 1 10); do n_rois=$((2**i))
    sem -j +0 python3 -m analysis.hierarchical "$n_rois" \
    --n_clusters "$1" \
    --mode "$2" \
    --metric "$3"\
    --task "$4"\
    --save_masks
  done
}

# run algorithm
run_hier "$n_clusters" "$mode" "$metric" "$task" # "$save_masks"
sem --wait

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'