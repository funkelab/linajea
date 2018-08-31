FROM gp_pylp
LABEL maintainer funkej@janelia.hhmi.org

# install jupyter and nyroglancer

RUN pip install --upgrade --ignore-installed notebook

ENV NYROGLANCER_ROOT=/src/nyroglancer
ENV NYROGLANCER_REPOSITORY=https://github.com/funkey/nyroglancer
ENV NYROGLANCER_REVISION=d621eadb31d17e4580ca5cb1a39cd921f8a1c0ea

WORKDIR ${NYROGLANCER_ROOT}
RUN git clone ${NYROGLANCER_REPOSITORY} . && \
    git checkout ${NYROGLANCER_REVISION}
ENV PYTHONPATH ${NYROGLANCER_ROOT}:$PYTHONPATH

ENV NEUROGLANCER_ROOT=/src/neuroglancer
ENV NEUROGLANCER_REPOSITORY=https://github.com/google/neuroglancer
ENV NEUROGLANCER_REVISION=253a5cc

WORKDIR ${NEUROGLANCER_ROOT}
RUN git clone ${NEUROGLANCER_REPOSITORY} . && \
    git checkout ${NEUROGLANCER_REVISION}
WORKDIR ${NEUROGLANCER_ROOT}/python
RUN python setup.py install

# install dependencies

# install python wrapper for KLB reader
ENV PYKLB_ROOT=/src/pyklb
ENV PYKLB_REPOSITORY=https://github.com/bhoeckendorf/pyklb
ENV PYKLB_REVISION=870874e26129411382323467d62038456bcb0385

RUN pip install wheel
WORKDIR ${PYKLB_ROOT}
RUN git clone ${PYKLB_REPOSITORY} . && \
    git checkout ${PYKLB_REVISION}
RUN python setup.py bdist_wheel
ENV PYTHONPATH=${PYKLB_ROOT}/build/lib.linux-x86_64-2.7:${PYTHONPATH}
RUN cp build/lib/libklb.so /lib

ENV DAISY_ROOT=/src/daisy
ENV DAISY_REPOSITORY=https://github.com/funkey/daisy
ENV DAISY_REVISION=9c89b3ef6dac48e370df852b9b9a60e31df1125c

WORKDIR ${DAISY_ROOT}
RUN git clone ${DAISY_REPOSITORY} . && \
    git checkout ${DAISY_REVISION}
ENV PYTHONPATH ${DAISY_ROOT}:$PYTHONPATH

RUN pip install zarr networkx pymongo
RUN pip install dask distributed --upgrade

# install linajea

# assumes that linajea package directory is in build context (the complementary
# Makefile ensures that)
ADD linajea /src/linajea/linajea
ADD requirements.txt /src/linajea/requirements.txt
ADD setup.py /src/linajea/setup.py
WORKDIR /src/linajea
RUN python setup.py install
