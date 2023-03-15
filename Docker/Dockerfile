FROM nvcr.io/nvidia/pytorch:22.05-py3

RUN rm -f /etc/apt/sources.list.d/*
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

RUN mkdir -p /var/run/sshd
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		autoconf \
		automake \
		autotools-dev \
		build-essential \
		ca-certificates \
		curl \
		daemontools \
		ibverbs-providers \
		libibverbs1 \
		libkrb5-dev \
		librdmacm1 \
		libssl-dev \
		libtool \
        libfabric-dev \
        libibverbs-dev \
        gfortran \
		git \
		krb5-user \
		cmake \
		g++ \
		make \
		openssh-client \
		openssh-server \
		pkg-config \
		wget \
		nfs-common \
		unattended-upgrades \
	&& unattended-upgrade \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm /etc/ssh/ssh_host_ecdsa_key \
	&& rm /etc/ssh/ssh_host_ed25519_key \
	&& rm /etc/ssh/ssh_host_rsa_key

#### DOWNLOAD AND INSTALL MPICH AND MPI4PY ####
WORKDIR /mpich
# Source is available at http://www.mpich.org/static/downloads/
# See installation guide of target MPICH version
# Ex: https://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2-installguide.pdf
# These options are passed to the steps below
ARG MPICH_VERSION="3.3"
ARG MPICH_CONFIGURE_OPTIONS="--prefix=/mpich/install --disable-wrapper-rpath"
ARG MPICH_MAKE_OPTIONS="-j 4"
RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
      && tar xfz mpich-${MPICH_VERSION}.tar.gz  --strip-components=1 \
      && ./configure ${MPICH_CONFIGURE_OPTIONS} \
      && make install ${MPICH_MAKE_OPTIONS}
ENV PATH $PATH:/mpich/install/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/mpich/install/lib
RUN pip install mpi4py


COPY dockerfile_scripts /tmp/det_dockerfile_scripts

# You can install your own packages as needed here.
# RUN pip install ...

# We uninstall these packages after installing. This ensures that we can
# successfully install these packages into containers running as non-root.
# `pip` does not uninstall dependencies, so we still have all the dependencies
# installed.
RUN python -m pip install determined && python -m pip uninstall -y determined

RUN /tmp/det_dockerfile_scripts/add_det_nobody_user.sh

RUN /tmp/det_dockerfile_scripts/install_libnss_determined.sh

RUN rm -r /tmp/*


RUN apt-get install cmake libhdf5-dev -y
RUN pip install numpy h5py larcv hydra-core==1.1 mpi4py
