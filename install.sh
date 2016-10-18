#!/bin/bash

# Install some python packages
pip install pandas
pip install numpy
pip install sklearn
pip install matplotlib
pip install -U imbalanced-learn
pip install orange
pip install graphviz
pip install xgboost
pip install rpy2
pip install pylatex

# For bayesian optimization: download source and install it
git clone https://github.com/fmfn/BayesianOptimization.git
cd BayesianOptimization-master
sudo python setup.py install
cd ..

# Special care needed for C45Learner from Orange
wget http://www.rulequest.com/Personal/c4.5r8.tar.gz
tar -xvzf rc4.5r8.tar.gz
cd R8/Src
wget https://github.com/biolab/orange/blob/master/Orange/orng/buildC45.py
wget https://github.com/biolab/orange/blob/master/Orange/orng/ensemble.c
sudo python buildC45.py
cd ..
cd ..

# Install some R packages
wget https://cran.r-project.org/src/contrib/randomForest_4.6-12.tar.gz
tar -xvzf randomForest_4.6-12.tar.gz
sudo R -e 'install.packages("'$(pwd)'/randomForest", repos=NULL, type="source")'
wget https://cran.r-project.org/src/contrib/inTrees_1.1.tar.gz
tar -xvzf inTrees_1.1.tar.gz
sudo R -e 'install.packages("'$(pwd)'/inTrees", repos=NULL, type="source")'


# sudo cp matplotlibrc /users/givdwiel/.local/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc



