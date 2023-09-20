#!/bin/bash

path=$1
name=$2

cd "${path}" || exit
n=$(awk -F '\t' '{print $1}' "${name}" | sort -n | uniq -c | wc -l)
svinet -file "${name}" -n "${n}" -k "${n}" -findk

# function to parrallelize
# run_findk() {
#   sem -j +0 svinet -file "${1}" -n "${2}" -k "${2}" -findk
# }

# pattern="*-mmsb-findk"
# n_runs=$(find . -maxdepth 1 -name "$pattern" -type d | wc -l)
# if ((seeds > n_runs)); then
  # find maximum seed of previous runs
  # max=0
  # runs=$(find . -maxdepth 1 -name "${pattern}" -type d)
  # for r in $runs; do
  #   x="$(cut -d '-' -f 4 <<<"${r}" | tr -dc '0-9')"
  #   if ((x > max)); then max=$x; fi
  # done
  # create loop over new seeds
  # sequence=$(seq $((max + 1)) 1 $((seeds)))
  # run algorithm
 #  run_svinet "${name}" "${n}" "${k}" "${sequence}" "${thresh}" "${sample}"
#   sem --wait
# else
 #  echo "${n_runs} runs found, skipping"
# fi

# printf '\n\n**************************************************************************\n'
# echo Done!
# printf '**************************************************************************\n\n'
