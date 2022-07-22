Linajea
=========

Publications
--------------
 - TODO link to nbt paper
 - TODO link to miccai paper

![Linajea](./README.assets/pipeline.png "Linajea Pipeline")

This is the main software repository for the linajea cell tracking project.
Includes tools and infrastructure for running a pipeline that starts from light
sheet data and ends in extracted cell lineage tracks.


Installation
--------------
```
conda create --name linajea
pip install numpy cython
pip install -r requirements.txt
conda install -c funkey pylp
pip install -e .
```

Versioning
------------
The main branch contains the current version of the code. New features and
bugfixes will be developed in separate branches before being merged into main.
The experiments in the NBT paper have been conducted with v1.3, the
experiments in the MICCAI paper with v1.4 (see tags). For the public release
we refactored major parts of the code, breaking backwards compatibility.
A separate repository (https://github.com/linajea/linajea_experiments) contains
all the scripts necessary to replicate the results, using the appropriate
release.


Contributing
--------------
If you make any improvements to the software, please fork, create a new branch
named descriptively for the feature you are upgrading or bug you are fixing,
commit your changes to that branch, and create a pull request asking for
permission to merge. Help is always appreciated! 


Other
------
Many of the modules in this project have their own README to provide more
information. If you can't find the answers you need there or in the code
documentation, feel free to contact us!
