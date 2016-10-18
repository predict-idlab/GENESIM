### GENESIM: GENetic Extraction of a Single, Interpretable Model

This repository contains an innovative algorithm that constructs an ensemble using well-known decision tree induction algorithms such as CART, C4.5, QUEST and GUIDE combined with bagging and boosting. Then, this ensemble is converted to a single, interpretable decision tree in a genetic fashion. For a certain number of iterations, random pairs of decision trees are merged together by first converting them to sets of k-dimensional hyperplanes and then calculating the intersection of these two sets (a classic problem from computational geometry). Moreover, in each iteration, an individual is mutated with a certain probabibility. After these iterations, the accuracy on a validation set is measured for each of the decision trees in the population and the one with the highest accuracy (and lowest number of nodes in case of a tie) is returned. Example.py has run code for all implemented algorithms and returns their average predictive performance, computational complexity and model complexity on a number of dataset

## Dependencies

An install.sh script is provided that will install all required dependencies

## Decision Tree Induction Algorithm Wrappers

A wrapper is written around [Orange C4.5](http://docs.orange.biolab.si/2/reference/rst/Orange.classification.tree.html#Orange.classification.tree.C45Learner), [sklearn CART](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [GUIDE](https://www.stat.wisc.edu/~loh/guide.html) and [QUEST](https://www.stat.wisc.edu/~loh/quest.html). The returned object is a Decision Tree, which can be found in `decisiontree.py`. Moreover, different methods are available on this decision tree: classify new, unknown samples; visualise the tree; export it to string, JSON and DOT; etc.

## Ensemble Technique Wrappers

A wrapper is written around the well-known state-of-the-art ensemble techniques [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html) and [Random Forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## New dataset

A new dataset can easily be plugged in into the benchmark. For this, a `load_dataset()` function must be written in `load_datasets.py`