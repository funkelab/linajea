FROM funkey/gunpowder:v0.3.2
LABEL maintainer funkej@janelia.hhmi.org

ENV PYLP_ROOT=/src/pylp
ENV PYLP_REPOSITORY=https://github.com/funkey/pylp
ENV PYLP_REVISION=2d52b5302002cb764c8acaf242998f19f85494f7

ADD gurobi /opt/gurobi
WORKDIR ${PYLP_ROOT} 
RUN git clone ${PYLP_REPOSITORY} . && \
    git checkout ${PYLP_REVISION} && \
    git submodule update --init
RUN GUROBI_ROOT_DIR=/opt/gurobi python setup.py install
