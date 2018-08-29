FROM gp_pylp
LABEL maintainer funkej@janelia.hhmi.org

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
