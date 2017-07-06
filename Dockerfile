# Use an official Python runtime as a base image
FROM python:2.7-slim

# Installing some command line tools, required to build all dependencies
RUN apt-get update
RUN apt-get -y install libc-dev
RUN apt-get -y install gcc
RUN apt-get -y install g++
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get -y install gfortran
RUN apt-get -y install build-essential
RUN apt-get -y install r-base
RUN apt-get -y install libcurl4-openssl-dev
RUN apt-get -y install tk
RUN apt-get -y install libcurl4-gnutls-dev
RUN apt-get -y install libssl-dev

# Clone the repo, set working dir
RUN git clone https://github.com/GillesVandewiele/GENESIM-1
WORKDIR /GENESIM-1

# Install the required python libraries
RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install -U imbalanced-learn
RUN pip install graphviz
RUN pip install xgboost
RUN pip install rpy2
RUN pip install pylatex
RUN pip install orange
RUN pip install bayesian-optimization

# Install R 3.3.2
RUN wget https://cran.rstudio.com/src/base/R-3/R-3.3.2.tar.gz
RUN tar -xvzf R-3.3.2.tar.gz
RUN cd R-3.3.2 && ./configure --with-readline=no --with-x=no && make && make install

# Special care needed for C45Learner from Orange
RUN wget http://www.rulequest.com/Personal/c4.5r8.tar.gz
RUN tar -xvzf c4.5r8.tar.gz
RUN cd R8/Src && wget https://raw.githubusercontent.com/biolab/orange/master/Orange/orng/buildC45.py && wget https://raw.githubusercontent.com/biolab/orange/master/Orange/orng/ensemble.c && python buildC45.py

# Install some R packages
RUN wget https://cran.r-project.org/src/contrib/randomForest_4.6-12.tar.gz
RUN tar -xvzf randomForest_4.6-12.tar.gz
RUN R -e 'install.packages("'$(pwd)'/randomForest", repos=NULL, type="source")'
RUN wget https://cran.r-project.org/src/contrib/inTrees_1.1.tar.gz
RUN tar -xvzf inTrees_1.1.tar.gz
RUN R -e 'install.packages("devtools", repos="http://cran.us.r-project.org")'
RUN R -e 'library(devtools); install("'$(pwd)'/inTrees", dependencies=TRUE)'

CMD ["python", "example.py"]
