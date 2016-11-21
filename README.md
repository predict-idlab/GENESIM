# GENESIM: GENetic Extraction of a Single, Interpretable Model

This repository contains an innovative algorithm that constructs an ensemble using well-known decision tree induction algorithms such as CART, C4.5, QUEST and GUIDE combined with bagging and boosting. Then, this ensemble is converted to a single, interpretable decision tree in a genetic fashion. For a certain number of iterations, random pairs of decision trees are merged together by first converting them to sets of k-dimensional hyperplanes and then calculating the intersection of these two sets (a classic problem from computational geometry). Moreover, in each iteration, an individual is mutated with a certain probabibility. After these iterations, the accuracy on a validation set is measured for each of the decision trees in the population and the one with the highest accuracy (and lowest number of nodes in case of a tie) is returned. Example.py has run code for all implemented algorithms and returns their average predictive performance, computational complexity and model complexity on a number of dataset

## Dependencies

An install.sh script is provided that will install all required dependencies

## Documentation

A nicely looking documentation page is available in the doc/ directory. Download the complete directory and open index.html

## Decision Tree Induction Algorithm Wrappers

A wrapper is written around [Orange C4.5](http://docs.orange.biolab.si/2/reference/rst/Orange.classification.tree.html#Orange.classification.tree.C45Learner), [sklearn CART](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [GUIDE](https://www.stat.wisc.edu/~loh/guide.html) and [QUEST](https://www.stat.wisc.edu/~loh/quest.html). The returned object is a Decision Tree, which can be found in `decisiontree.py`. Moreover, different methods are available on this decision tree: classify new, unknown samples; visualise the tree; export it to string, JSON and DOT; etc.

## Ensemble Technique Wrappers

A wrapper is written around the well-known state-of-the-art ensemble techniques [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html) and [Random Forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Similar techniques

A wrapper written around the R package [inTrees](https://arxiv.org/abs/1408.5456) and an implementation of [ISM](https://lirias.kuleuven.be/handle/123456789/146229) can be found in the constructors package.

## New dataset

A new dataset can easily be plugged in into the benchmark. For this, a `load_dataset()` function must be written in `load_datasets.py`

## Contact

You can contact me at givdwiel.vandewiele at ugent.be for any questions, proposals or if you wish to contribute.

## Referring

Please refer to my work when you use it. A reference to this github or to the following (yet unpublished) paper:

`
@article{Vandewiele,
abstract = {Models obtained by decision tree induction techniques excel in being interpretable. However, they can be prone to overfitting, which results in a low predictive per-formance. Ensemble techniques are able to achieve a higher accuracy. However, this comes at a cost of losing interpretability of the resulting model. This makes ensemble techniques impractical in applications where decision support, instead of decision making, is crucial. To bridge this gap, we present the GENESIM algorithm that transforms an ensemble of decision trees to a single decision tree with an enhanced predictive performance by using a genetic algorithm. We compared GENESIM to prevalent decision tree induction and ensemble techniques using twelve publicly available data sets. The results show that GENESIM achieves a better predictive performance on most of these data sets than decision tree induction techniques and a predictive performance in the same order of magnitude as the ensemble techniques. Moreover, the resulting model of GENESIM has a very low complexity, making it very interpretable, in contrast to ensemble techniques.},
author = {Vandewiele, Gilles and Janssens, Olivier and Ongenae, Femke and {De Turck}, Filip and Hoecke, Sofie Van},
title = {{GENESIM: genetic extraction of a single, interpretable model}},
year={2016},
note={available at; \url{https://github.com/IBCNServices/GENESIM}}
}
`

