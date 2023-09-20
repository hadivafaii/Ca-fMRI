#!/bin/bash

nn=${1}
ll=${2}
mode=${3}
pp=${4}
kk=${5}
key=${6:-"all"}
desc=${7:-"all"}
task=${8:-"rest"}
metric=${9:-"pearson"}
match_metric=${10-"euclidean"}
match_using=${11-"gam"}
graph_type=${12-"real"}

if [ -z "${nn}" ]; then
  read -rp "please enter your favorite num ROIs: " nn
fi

cd ..

run_svinet () {
  python3 -m analysis.svinet \
  "${1}" "${2}" \
  --k "${3}" \
  --p "${4}" \
  --mode "${5}" \
  --desc "${6}" \
  --task "${7}" \
  --key "${8}" \
  --metric "${9}" \
  --match_metric "${10}" \
  --match_using "${11}" \
  --graph_type "${12}"
}

# run algorithm
run_svinet "${nn}" "${ll}" "${kk}" "${pp}" \
"${mode}" "${desc}" "${task}" "${key}" "${metric}" \
"${match_metric}" "${match_using}" "${graph_type}"
sem --wait

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'