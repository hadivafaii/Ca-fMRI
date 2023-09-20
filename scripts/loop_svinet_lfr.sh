#!/bin/bash


kind=$1
seeds=${2:-10}
k=${3:-"read"}
maxiter=${4:-0}
thresh=${5:-0.5}
sample=${6:-"link"}
ws_dir=${7:-"Documents/workspaces/lfr"}

if [ -z "$kind" ]; then
  read -rp "Enter what kind of LFR you want (options: 'bo' 'wo' 'h'): " kind
fi

if [[ "$kind" == "bo" ]]; then
  kind="binary_overlapping"
elif [[ "$kind" == "wo" ]]; then
  kind="weighted_overlapping"
elif [[ "$kind" == "h" ]]; then
  kind="hierarchical"
else
  echo "invalid value, available options: 'bo' 'wo' 'h'"
  exit
fi

root="${HOME}/${ws_dir}/${kind}/results"
dirs=$(find "$root" -mindepth 1 -type d)

pattern="network.dat"
for d in $dirs; do
  files=$(find "$d" -name "$pattern" -type f)
  for f in $files; do
    path=$(dirname "${f}")
    name=$(basename "${f}")
    if [[ "$k" == "read" ]]; then
      _k=$(<"${path}/true_k.dat")
    else
      _k=$((k))
    fi
    ./_svinet.sh "${path}" "${name}" "${_k}" \
    "${seeds}" "${maxiter}" "${thresh}" "${sample}"
  done
done