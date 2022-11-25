Linajea
=========

Publications
--------------
 - [Automated reconstruction of whole-embryo cell lineages by learning from sparse annotations (nature biotechnology)](https://www.nature.com/articles/s41587-022-01427-7)
 - [Tracking by weakly-supervised learning and graph optimization for whole-embryo C. elegans lineages (arxiv/MICCAI2022)](https://arxiv.org/abs/2208.11467)

![Linajea](./README.assets/pipeline.png "Linajea Pipeline")

This is the main software repository for the linajea cell tracking project.
It includes tools and infrastructure for running a pipeline that starts from 3d+time light sheet data and ends in extracted cell lineage tracks.


Installation
--------------
```
git clone https://github.com/funkelab/linajea.git
cd linajea
conda create --name linajea python pytorch pylp -c pytorch -c funkey
conda activate linajea
pip install numpy cython jupyter
pip install -r requirements.txt
pip install -e .
conda install ipykernel # for the example jupyter notebooks
```

Versioning
------------
The main branch contains the current version of the code. New features and bugfixes will be developed in separate branches before being merged into main.
The experiments in the Nature Biotechnology paper have been conducted with v1.3, the experiments in the MICCAI paper with v1.4 (see tags). For the public release we refactored major parts of the code, breaking backwards compatibility.
A separate repository (https://github.com/linajea/linajea_experiments) contains all the scripts necessary to replicate the paper results, using the appropriate release.


Use
---
Have a look at the jupyter notebook [examples](examples) or look at the [run scripts](run_scripts) directly.


Contributing
--------------
If you make any improvements to the software, please fork, create a new branch named descriptively for the feature you are upgrading or bug you are fixing, commit your changes to that branch, and create a pull request asking for permission to merge.
Help is always appreciated!


Other
------
If you have any questions and can't find the answers you need in the examples or in the code documentation, feel free to contact us!
