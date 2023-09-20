#!/bin/bash

nn=${1}
ll=${2}
mode=${3}
pp=${4}
kk=${5:-"-1"}
desc=${6:-"all"}
task=${7:-"rest"}
metric=${8:-"pearson"}
match_metric=${9-"euclidean"}
match_using=${10-"gam"}
graph_type=${11-"real"}

if [ -z "${nn}" ]; then
  read -rp "please enter your favorite num ROIs: " nn
fi

cd ..

run_group () {
  python3 -m analysis.group \
  "${1}" "${2}" \
  --k "${3}" \
  --p "${4}" \
  --mode "${5}" \
  --desc "${6}" \
  --task "${7}" \
  --metric "${8}" \
  --match_metric "${9}" \
  --match_using "${10}" \
  --graph_type "${11}"
}

# run algorithm
run_group "${nn}" "${ll}" "${kk}" "${pp}" \
"${mode}" "${desc}" "${task}" "${metric}" \
"${match_metric}" "${match_using}" "${graph_type}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'