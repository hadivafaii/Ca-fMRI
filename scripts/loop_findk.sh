#!/bin/bash


nn=${1}
ll=${2:-3}
pp=${3:-"p*.txt"}
mode=${4:-"*"}
task=${5:-"*"}
metric=${6:-"pearson"}
key=${7-"*"}
ws_dir=${8:-"Documents/workspaces/svinet"}

if [ -z "${nn}" ]; then
  read -rp "please enter n ROIs: " nn
fi

if [[ $pp != *.txt ]]; then
  pp="${pp}.txt"
fi

root="${HOME}/${ws_dir}"
pattern="n-${nn}*${ll}_${mode}_${task}_${metric}_${key}"
dirs=$(find "${root}" -maxdepth 3 -name "${pattern}" -type d)

for d in $dirs; do
  files=$(find "$d" -name "${pp}" -type f)
  for f in $files; do
    path=$(dirname "${f}")
    name=$(basename "${f}")
    ./_findk.sh "${path}" "${name}"
  done
done