# Welcome!

This repository contains code for reproducing results in the paper "*Multimodal measures of spontaneous brain activity reveal both common and divergent patterns of cortical functional organization*" (DOI: [10.21203/rs.3.rs-2823802/v1](https://www.researchsquare.com/article/rs-2823802/v1)).

## Introdution
We used SVINET [1], a publicly available open-source software, to perform overlapping network analysis. This algorithm was introduced in 2013 and has since been thoroughly tested and validated in various types of networks, including human fMRI networks, by our team [2] as well as others [3]. Additionally, we make use of Python-based scientific computing libraries such as Numpy and SciPy for basic operations such as computing Pearson correlation, cosine similarity, averaging, etc.

## Quick start guide
The following object classes were used in the main analysis. Full documentation to follow.
- Network class (```analysis/network.py```) to generate adjacency matrices and save binarized graphs. Corresponding script: ```scripts/run_network.sh```
- SVINET class (```analysis/svinet.py```) to load overlapping communities (output from the main algorithm) and align them across random seeds for each individual run. Corresponding script: ```scripts/run_svinet.sh```
- Group class (```analysis/group.py```) to perform group analysis. Combines all runs by aligning and averaging them which yields the group results. Corresponding script: ```scripts/run_group.sh```
- Bootstrap class (```analysis/bootstrap.py```) to perform various statistical analysis on the data. Corresponding script: ```scripts/run_bootstrap.sh```
- LFR analysis (```analysis/lfr.py```) to perform community analysis on synthetic LFR graphs with known ground truths [4]. Corresponding script: ```scripts/run_lfr.sh```

Please see ```scripts/examples.txt``` for a demonstration of how to execute these scripts.

## Additional modules
- Allen class (```register/atlas.py```) is a wrapper around the [Allen SDK](https://allensdk.readthedocs.io/en/latest/) package. Used to extract brain masks, templates, and other type of information available within the CCFv3 common space [5].
- Parcellation class (```register/parcellation.py```) creates a spatially homogeneous parcellation of the mouse cortex. Please refer to ROI definition section in Methods.
- Register class (```register/register.py```) is used to co-register individual mouse brains to the CCFv3 common space. Corresponding script: ```scripts/run_register.sh```

## Installation guide
The present codebase is built as a wrapper around SVINET and uses standard Python libraries. It has been tested on Ubuntu 22.04.2 LTS. For installation guide, please refer to the original sources:

- SVINET (master branch, v2015): https://github.com/premgopalan/svinet
- Numpy (v 1.21.2): https://numpy.org/
- SciPy (v 1.7.0): https://scipy.org/
- Statsmodels (v 0.13.5): https://www.statsmodels.org/stable/index.html
- Allen SDK (v 2.12.3): https://allensdk.readthedocs.io/en/latest/index.html
- LFR graphs: https://www.santofortunato.net/resources
    - package 1, undirected and unweighted graphs with overlapping communities

Typical install time for these packages is in the order of several minutes.

## Documentation
Coming soon 🚧

## References:
1. Gopalan, Prem K., and David M. Blei. "Efficient discovery of overlapping communities in massive networks." PNAS 110, no. 36 (2013): 14534-14539.
2. Najafi, Mahshid, et al. "Overlapping communities reveal rich structure in large-scale brain networks during rest and task conditions." Neuroimage 135 (2016): 92-106.
3. Cookson, Savannah L., and Mark D’Esposito. "Evaluating the reliability, validity, and utility of overlapping networks: Implications for network theories of cognition." Human Brain Mapping 44.3 (2023): 1030-1045.
4. Lancichinetti, A. and Fortunato, S., 2009. "Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities". Physical Review E, 80(1), p.016118.
5. Wang, Q., et al. "The Allen mouse brain common coordinate framework: a 3D reference atlas." Cell 181.4 (2020): 936-953.
