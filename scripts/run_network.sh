#!/bin/bash

nn=${1}
ll=${2}
mode=${3}
desc=${4:-"all"}
task=${5:-"rest"}
metric=${6:-"pearson"}
key=${7:-"all"}
full=${8-false}

if [ -z "${nn}" ]; then
  read -rp "please enter your favorite num ROIs: " nn
fi

cd ..

run_net () {
  if ${8}; then
    python3 -m analysis.network \
    "${1}" "${2}" \
    --mode "${3}" \
    --desc "${4}" \
    --task "${5}" \
    --metric "${6}" \
    --key "${7}" \
    --full
  else
    python3 -m analysis.network \
    "${1}" "${2}" \
    --mode "${3}" \
    --desc "${4}" \
    --task "${5}" \
    --metric "${6}" \
    --key "${7}"
  fi
}

# run algorithm
run_net "${nn}" "${ll}" "${mode}" "${desc}" \
  "${task}" "${metric}" "${key}" "${full}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'