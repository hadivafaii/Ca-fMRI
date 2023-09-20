#!/bin/bash

path=$1
name=$2
k=${3:-7}
seeds=${4:-500}
maxiter=${5:-0}
thresh=${6:-0.5}
sample=${7:-"link"}

# go to the dir where network txt file is saved
cd "${path}" || exit

# count number of unique entries to determine num nodes
n=$(awk -F '\t' '{print $1}' "${name}" | sort -n | uniq -c | wc -l)

# reduce number of seeds for config graphs
if [[ "${path}" =~ "/config" ]]; then
  seeds=$((seeds / 10))
fi

# Max iter = 0: use early stopping
# Max iter > 0: fixed num iterations
re="^[0-9]+\d*$"
if ! [[ ${maxiter} =~ $re ]]; then
  maxiter=0
fi

printf '\n****************************************************************************\n'
echo "${path#*svinet}"
printf "name: %s,\nn: %d,\nk: %d,\nnum seeds: %d,\nmax iter: %d,\n" "${name}" "${n}" "${k}" "${seeds}" "${maxiter}"
printf '****************************************************************************\n\n'

# function to parrallelize
run_svinet() {
  for seed in ${4}; do
    if [[ "${7}" == "link" ]]; then
      if [ "${5}" -eq "0" ]; then
        sem -j +0 svinet -file "${1}" -n "${2}" -k "${3}" -seed "${seed}" -link-thresh "${6}" -link-sampling
      else
        sem -j +0 svinet -file "${1}" -n "${2}" -k "${3}" -seed "${seed}" -link-thresh "${6}" -link-sampling -no-stop -max-iterations "${5}"
      fi
    elif [[ "${7}" == "rnode" ]]; then
      if [ "${5}" -eq "0" ]; then
        sem -j +0 svinet -file "${1}" -n "${2}" -k "${3}" -seed "${seed}" -link-thresh "${6}" -rnode
      else
        sem -j +0 svinet -file "${1}" -n "${2}" -k "${3}" -seed "${seed}" -link-thresh "${6}" -rnode -no-stop -max-iterations "${5}"
      fi
    elif [[ "${7}" == "rpair" ]]; then
      if [ "${5}" -eq "0" ]; then
        sem -j +0 svinet -file "${1}" -n "${2}" -k "${3}" -seed "${seed}" -link-thresh "${6}" -rpair
      else
        sem -j +0 svinet -file "${1}" -n "${2}" -k "${3}" -seed "${seed}" -link-thresh "${6}" -rpair -no-stop -max-iterations "${5}"
      fi
    fi
  done
}

pattern="*-k${k}-*"
n_runs=$(find . -maxdepth 1 -name "$pattern" -type d | wc -l)
if ((seeds > n_runs)); then
  # find maximum seed of previous runs
  max=0
  runs=$(find . -maxdepth 1 -name "${pattern}" -type d)
  for r in $runs; do
    x="$(cut -d '-' -f 4 <<<"${r}" | tr -dc '0-9')"
    if ((x > max)); then max=$x; fi
  done
  # create loop over new seeds
  sequence=$(seq $((max + 1)) 1 $((seeds)))
  # run algorithm
  run_svinet "${name}" "${n}" "${k}" "${sequence}" "${maxiter}" "${thresh}" "${sample}"
  sem --wait
else
  echo "${n_runs} runs found, skipping"
fi

printf '\n\n****************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '****************************************************************************\n\n'
