#!/bin/bash

nn=${1}
ll=${2:-3}
kk=${3:-"-1"}
pp=${4:-"p20-sample"}
num=${5:-1000000}
njobs=${6:-"-1"}
mode=${7:-"ca2"}
task=${8:-"all"}
metric=${9:-"pearson"}
match_metric=${10-"euclidean"}
match_using=${11-"gam"}
graph_type=${12-"real"}
scipy=${13-true}
group=${14-true}

if [ -z "${nn}" ]; then
  read -rp "please enter your favorite num ROIs: " nn
fi

cd ..

# make function to parrallelize
run_bootstrap () {
  if ${13}; then
    if ${14}; then
      python3 -m analysis.bootstrap \
      "${1}" "${2}" \
      --k "${3}" \
      --p "${4}" \
      --num "${5}" \
      --njobs "${6}" \
      --mode "${7}" \
      --task "${8}" \
      --metric "${9}" \
      --match_metric "${10}" \
      --match_using "${11}" \
      --graph_type "${12}" \
      --scipy \
      --group
    else
      python3 -m analysis.bootstrap \
      "${1}" "${2}" \
      --k "${3}" \
      --p "${4}" \
      --num "${5}" \
      --njobs "${6}" \
      --mode "${7}" \
      --task "${8}" \
      --metric "${9}" \
      --match_metric "${10}" \
      --match_using "${11}" \
      --graph_type "${12}" \
      --scipy
    fi
  else
    if ${14}; then
      python3 -m analysis.bootstrap \
      "${1}" "${2}" \
      --k "${3}" \
      --p "${4}" \
      --num "${5}" \
      --njobs "${6}" \
      --mode "${7}" \
      --task "${8}" \
      --metric "${9}" \
      --match_metric "${10}" \
      --match_using "${11}" \
      --graph_type "${12}" \
      --group
    else
      python3 -m analysis.bootstrap \
      "${1}" "${2}" \
      --k "${3}" \
      --p "${4}" \
      --num "${5}" \
      --njobs "${6}" \
      --mode "${7}" \
      --task "${8}" \
      --metric "${9}" \
      --match_metric "${10}" \
      --match_using "${11}" \
      --graph_type "${12}"
    fi
  fi
}

# run algorithm
run_bootstrap "${nn}" "${ll}" "${kk}" "${pp}" "${num}" "${njobs}" \
"${mode}" "${task}" "${metric}" "${match_metric}" "${match_using}" "${graph_type}" \
"${scipy}" "${group}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'