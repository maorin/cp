FROM ubuntu:latest



RUN apt-get -qq update && apt-get -qq -y install gcc curl bzip2 build-essential \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    /apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH


RUN conda install -y \
    h5py \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    theano \
    jupyter \
    tensorflow \
    keras \
    bcolz \
    && conda clean --yes --tarballs --packages --source-cache

RUN apt-get install locales -y
RUN locale-gen en_US.UTF-8

ADD . /code
WORKDIR /code

RUN pip install -r requirements.txt -i https://pypi.douban.com/simple


CMD python cp/doraemon.py