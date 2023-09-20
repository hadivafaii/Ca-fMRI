#!/bin/bash

nn=${1}
ll=${2}
mode=${3}
pp=${4:-"p*.txt"}
kk=${5:-7}
seeds=${6:-500}
desc=${7-"[a-z-]+"}
bp=${8-"[0-9.,]+"}
# "[0-9.,]+": covers all
# "(0.01,0.5|0.5,5.0)": | = or
task=${9:-"(rest|led)"}
metric=${10:-"pearson"}
key=${11-"sub-SLC[-_0-9a-z]*"}
maxiter=${12:-0}
thresh=${13:-0.5}
sample=${14:-"link"}
ws_dir=${15:-"Documents/workspaces/svinet"}

if [ -z "${nn}" ]; then
  read -rp "please enter n ROIs: " nn
fi
if [[ $pp != *.txt ]]; then
  pp="${pp}.txt"
fi

root="${HOME}/${ws_dir}"
r=".*n-${nn}[*]${ll}_${metric}_${task}_${mode}-${desc}_bp[(]${bp}[)]_${key}"
dirs=$(find "${root}" -maxdepth 2 -regextype posix-egrep -regex "${r}" -type d)

for d in $dirs; do
  files=$(find "$d" -name "${pp}" -type f)
  for f in $files; do
    path=$(dirname "${f}")
    name=$(basename "${f}")
    ./_svinet.sh "${path}" "${name}" "${kk}" \
    "${seeds}" "${maxiter}" "${thresh}" "${sample}"
  done
done
