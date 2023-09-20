#!/bin/bash


kind=$1
ws_dir=${2:-"Documents/workspaces/lfr"}

if [ -z "$kind" ]; then
  read -rp "Enter what kind of LFR you want (options: 'bo' 'wo' 'h'): " kind
fi

if [[ "${kind}" == "bo" ]]; then
  kind="binary_overlapping"
elif [[ "${kind}" == "wo" ]]; then
  kind="weighted_overlapping"
elif [[ "${kind}" == "h" ]]; then
  kind="hierarchical"
else
  echo "invalid value, available options: 'bo' 'wo' 'h'"
  exit
fi

root="${HOME}/${ws_dir}/${kind}"
dirs=$(find "${root}" -name "results" -type d)

cd "${root}" || exit

printf '\n**************************************************************************\n'
echo "${root}"
printf '**************************************************************************\n\n\n'

pattern="flags.dat"
for d in $dirs; do
  files=$(find "${d}" -name "${pattern}" -type f)
  for f in $files; do
    path=$(dirname "${f}")
    name=$(basename "${f}")
    cp "${f}" "${root}/${name}"
    if [[ "${kind}" == "hierarchical" ]]; then
      ./hbenchmark -f flags.dat
      cp  network.dat community_first_level.dat community_second_level.dat "${path}"
    else
      ./benchmark -f flags.dat
      cp  network.dat community.dat statistics.dat "${path}"
    fi
  done
done

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'


: <<'END'
---------------------- How to run the program ----------------------
./benchmark [FLAG] [P]
----------------------

binary_overlapping:
----------------------
    -N		number of nodes
    -k		average degree
    -maxk		maximum degree
    -mu		mixing parameter
    -t1		minus exponent for the degree sequence
    -t2		minus exponent for the community size distribution
    -minc		minimum for the community sizes
    -maxc		maximum for the community sizes
    -on		number of overlapping nodes
    -om		number of memberships of the overlapping nodes
    -C              [average clustering coefficient]
----------------------

weighted_overlapping:
----------------------
    -N		number of nodes
    -k		average degree
    -maxk		maximum degree
    -mut		mixing parameter for the topology
    -muw		mixing parameter for the weights
    -beta		exponent for the weight distribution
    -t1		minus exponent for the degree sequence
    -t2		minus exponent for the community size distribution
    -minc		minimum for the community sizes
    -maxc		maximum for the community sizes
    -on		number of overlapping nodes
    -om		number of memberships of the overlapping nodes
    -C              [average clustering coefficient]
----------------------

hierarchical:
----------------------
    1.	-N              [number of nodes]
    2.	-k              [average degree]
    3.	-maxk           [maximum degree]
    4.	-t1             [minus exponent for the degree sequence]

    5.	-minc           [minimum for the micro community sizes]
    6.	-maxc           [maximum for the micro community sizes]
    7.	-on             [number of overlapping nodes (micro communities only)]
    8.	-om             [number of memberships of the overlapping nodes (micro communities only)]

    9.	-t2             [minus exponent for the community size distribution]

    10.	-minC           [minimum for the macro community size]
    11.	-maxC           [maximum for the macro community size]

    12.	-mu1            [mixing parameter for the macro communities (see below)]
    13.	-mu2            [mixing parameter for the micro communities (see below)]
----------------------
END