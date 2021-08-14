FROM gaetanlandreau/pytorch3d
RUN apt-get update && apt-get install -y \
    parallel \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /proj/shapmagn
COPY ./shapmagn/requirement.txt .
RUN pip install -r  requirement.txt
RUN pip install open3d --ignore-installed PyYAML
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
RUN apt-get update && apt-get -y install cmake protobuf-compiler && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update && apt-get -y install libeigen3-dev libboost-all-dev gifsicle
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.8/cmake-3.19.8-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake_318 \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake_318 \
      && rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake_318/bin:${PATH}"

ARG user=appuser
ARG group=appuser
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ${group} && useradd -u ${uid} -g ${group} -s /bin/sh ${user}
RUN mkdir /.cache
RUN mkdir /.local
RUN chmod 777 /.cache
RUN chmod 777 /.local
RUN chmod 777 /opt/conda/lib/python3.8/site-packages/
RUN mkdir /home/${user}
RUN chmod 777 /home/${user}


ENV PCL_PACKAGE_DIR="/proj/pcl"
RUN mkdir "$PCL_PACKAGE_DIR"
WORKDIR "$PCL_PACKAGE_DIR"
RUN apt-get update && apt-get -y install libflann-dev
RUN wget "https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.1.zip"
RUN unzip pcl-*.zip
RUN rm pcl-*.zip
RUN mv pcl-* pcl-lastest
RUN cd pcl-lastest && mkdir build && cd build
WORKDIR pcl-lastest/build
RUN cmake ..
RUN make -j $(python3 -c 'import multiprocessing as mp; print(int(mp.cpu_count()    * 1.5))')
RUN make install
RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
# Configure and compile
WORKDIR TEASER-plusplus
RUN mkdir build
WORKDIR build
RUN cmake ..
RUN make
# Generate doxygen documentation in doc/
RUN make doc
RUN make install
RUN cmake -DTEASERPP_PYTHON_VERSION=3.8 ..
RUN make teaserpp_python
WORKDIR python
RUN pip install .



USER ${user}
CMD /bin/bash