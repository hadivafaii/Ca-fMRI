#!/bin/bash

nn=${1}
ll=${2:-3}
cutoff=${3:-0}
thres=${4:-0.2}
anat=${5-false}
seeds=${6:-10}
space=${7:-"CCF"}
verbose=${8-true}

if [ -z "${nn}" ]; then
  read -rp "please enter your favorite num ROIs: " nn
fi

cd ..

run_register () {
  if ${7}; then
    if ${8}; then
      python3 -m register.register \
      "${1}" "${2}" \
      --cutoff "${3}" \
      --thres "${4}" \
      --seeds "${5}" \
      --space "${6}" \
      --anat \
      --verbose
    else
      python3 -m register.register \
      "${1}" "${2}" \
      --cutoff "${3}" \
      --thres "${4}" \
      --seeds "${5}" \
      --space "${6}" \
      --anat
    fi
  else
    if ${8}; then
      python3 -m register.register \
      "${1}" "${2}" \
      --cutoff "${3}" \
      --thres "${4}" \
      --seeds "${5}" \
      --space "${6}" \
      --verbose
    else
      python3 -m register.register \
      "${1}" "${2}" \
      --cutoff "${3}" \
      --thres "${4}" \
      --seeds "${5}" \
      --space "${6}"
    fi
  fi
}

# run algorithm
run_register "${nn}" "${ll}" "${cutoff}" "${thres}" \
"${seeds}" "${space}" "${anat}" "${verbose}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'