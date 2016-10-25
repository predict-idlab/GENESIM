"""
    Interpretable Single Model
    --------------------------

    Merges different decision trees in an ensemble together in a single, interpretable decision tree

    Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.

    Reference:
        Van Assche, Anneleen, and Hendrik Blockeel.
        "Seeing the forest through the trees: Learning a comprehensible model from an ensemble."
        European Conference on Machine Learning. Springer Berlin Heidelberg, 2007.
"""

from collections import Counter
import math

import numpy as np

import decisiontree


def _extract_tests(tree, _tests=set()):
    """
    Given a decision tree, extract all tests from the nodes

    :param tree: the decision tree to extract tests from (decisiontree.py)
    :param _tests: recursive parameter, don't touch
    :return: a set of possible tests (feature_label <= threshold_value); each entry is a tuple (label, value)
    """
    if tree.value is not None:
        _tests.add((tree.label, tree.value))
        _extract_tests(tree.left, _tests)
        _extract_tests(tree.right, _tests)
    return _tests


def _calculate_entropy(probabilities):
    """
    Calculate the entropy of given probabilities

    :param probabilities: a list of floats between [0, 1] (sum(probabilities) must be 1)
    :return: the entropy
    """
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def _get_most_occurring_class(data, class_label):
    """
    Get the most occurring class in a dataframe of data

    :param data: a pandas dataframe
    :param class_label: the column of the class labels
    :return: the most occurring class
    """
    return Counter(data[class_label].values.tolist()).most_common(1)[0][0]


def _calculate_prob(tree, label, value, prior_tests, negate=False):
    """
    Estimate the probabilities from a decision tree by propagating down from the root to the leaves

    :param tree: the decision tree to estimate the probabilities from
    :param label: the label of the test being evaluated
    :param value: the value of the test being evaluated
    :param prior_tests: tests that are already in the conjunctions
    :param negate: is it a negative or positive test
    :return: a vector of probabilities for each class
    """
    if tree.value is None:  # If the value is None, we're at a leaf, return a vector of probabilities
        return np.divide(list(map(float, list(tree.class_probabilities.values()))), float(sum(list(tree.class_probabilities.values()))))
    else:
        if (tree.label, tree.value) in prior_tests:
            # The test in the current node is already in the conjunction, take the correct path
            if prior_tests[(tree.label, tree.value)]:
                return _calculate_prob(tree.left, label, value, prior_tests, negate)
            else:
                return _calculate_prob(tree.right, label, value, prior_tests, negate)
        elif not (tree.label == label and tree.value == value):
            # The test of current node is not yet in conjunction and is not the test we're looking for
            # Keep propagating (but add weights (estimate how many times the test succeeds/fails))!
            samples_sum = sum(list(tree.class_probabilities.values()))
            if samples_sum == 0:
                left_fraction = 1.0
                right_fraction = 1.0
            else:
                left_fraction = sum(list(tree.left.class_probabilities.values())) / samples_sum
                right_fraction = sum(list(tree.right.class_probabilities.values())) / samples_sum

            return np.add(left_fraction * _calculate_prob(tree.left, label, value, prior_tests, negate),
                          right_fraction * _calculate_prob(tree.right, label, value, prior_tests, negate))
        elif not negate:
            # We found the test we are looking for
            # If negate is False, then it is a positive test and we take the left subtree
            return _calculate_prob(tree.left, label, value, prior_tests, negate)
        else:
            return _calculate_prob(tree.right, label, value, prior_tests, negate)


def _calculate_prob_dict(tree, label, value, prior_tests, negate=False):
    """
    Wrapper around calculate_prob, so we know which probability belongs to which class
    """
    return dict(zip(tree.class_probabilities.keys(), _calculate_prob(tree, label, value, prior_tests, negate)))


def ism(decision_trees, data, class_label, min_nr_samples=1, calc_fracs_from_ensemble=False):
    """
        Return a single decision tree from an ensemble of decision tree, using the normalized information gain as
        split criterion, estimated from the ensemble. This is a wrapper function around `constructors.ISM.build_dt_from_ensemble`,
        which first calculate the required parameters for this method.

    **Params**
    ----------
     - `decision_trees` (list of `decisiontree.DecisionTree` objects): the ensemble of decision trees to be merged

     - `data` (pandas DataFrame): the data frame with training data

     - `class_label` (string): the column identifier for the column with class labels in the data

     - `min_nr_samples` (int): pre-prune condition, stop searching if number of samples is smaller or equal than threshold

     - `calc_fracs_from_ensemble` (boolean): if `True`, the different probabilities are calculated using the ensemble. Else, the data is used

    **Returns**
    -----------
        a single decision tree based on the ensemble of decision trees
    """
    X = data.drop(class_label, axis=1).reset_index(drop=True)
    y = data[class_label].reset_index(drop=True)

    non_empty_decision_trees = []
    for tree in decision_trees:
        if tree.count_nodes() > 1: non_empty_decision_trees.append(tree)
    decision_trees = non_empty_decision_trees

    prior_entropy = 0
    tests = set()
    tests.clear()
    for dt in decision_trees:
        tests = tests | _extract_tests(dt, set())
        prior_entropy += _calculate_entropy(np.divide(list(dt.class_probabilities.values()),
                                                      sum(dt.class_probabilities.values())))
    prior_entropy /= len(decision_trees)

    combined_dt = build_dt_from_ensemble(decision_trees, data, class_label, tests, prior_entropy, {}, min_nr_samples,
                                         calc_fracs_from_ensemble)
    combined_dt.populate_samples(X, y)

    return combined_dt


def _add_reduce_by_key(A, B):
    """
    Reduces two dicts by key using add operator

    :param A: dict one
    :param B: dict two
    :return: a new dict, containing a of the values if the two dicts have the same key, else just the value
    """
    return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


def build_dt_from_ensemble(decision_trees, data, class_label, tests, prior_entropy, prior_tests={}, min_nr_samples=1,
                           calc_fracs_from_ensemble=False):
    """
    Given an ensemble of decision trees, build a single decision tree using estimates from the ensemble

    **Params**
    ----------
     - `decision_trees` (list of `decisiontree.DecisionTree` objects): the ensemble of decision trees to be merged

     - `data` (pandas DataFrame): the data frame with training data

     - `class_label` (string): the column identifier for the column with class labels in the data

     - `tests` (set of tuples): all possible tests (extracted from the ensemble)

     - `prior_entropy` (float): recursive parameter to calculate information gain

     - `prior_tests` (set of tuples): the tests that are already picked for our final decision tree

     - `min_nr_samples` (int): pre-prune condition, stop searching if number of samples is smaller or equal than threshold

     - `calc_fracs_from_ensemble` (boolean): if `True`, the different probabilities are calculated using the ensemble. Else, the data is used

    **Returns**
    -----------
        a single decision tree, calculated using information from the ensemble
    """
    # Pre-pruning conditions:
    #   - if the length of data is <= min_nr_samples
    #   - when we have no tests left
    #   - when there is only 1 unique class in the data left
    # print len(data), len(tests), np.unique(data[class_label].values)
    if len(data) > min_nr_samples and len(tests) > 0 and len(np.unique(data[class_label].values)) > 1:
        max_ig = 0
        best_pos_data, best_neg_data, best_pos_entropy, best_neg_entropy = [None]*4
        best_dt = decisiontree.DecisionTree()
        # Find the test that results in the maximum information gain
        for test in tests:
            pos_avg_probs, neg_avg_probs, pos_fraction, neg_fraction = {}, {}, 0.0, 0.0
            for dt in decision_trees:
                pos_prob_dict = _calculate_prob_dict(dt, test[0], test[1], prior_tests, False)
                neg_prob_dict = _calculate_prob_dict(dt, test[0], test[1], prior_tests, True)

                if not any(math.isnan(x) for x in pos_prob_dict.values()) and not any(math.isnan(x) for x in neg_prob_dict.values()):
                    pos_avg_probs = _add_reduce_by_key(pos_avg_probs, _calculate_prob_dict(dt, test[0], test[1], prior_tests, False))
                    neg_avg_probs = _add_reduce_by_key(neg_avg_probs, _calculate_prob_dict(dt, test[0], test[1], prior_tests, True))

                if calc_fracs_from_ensemble and len(data) > 0:
                    pos_fraction += float(len(dt.data[dt.data[test[0]] <= test[1]]))/len(dt.data)
                    neg_fraction += float(len(dt.data[dt.data[test[0]] > test[1]]))/len(dt.data)

            for key in pos_avg_probs:
                pos_avg_probs[key] /= len(decision_trees)
            for key in neg_avg_probs:
                neg_avg_probs[key] /= len(decision_trees)

            if calc_fracs_from_ensemble:
                pos_fraction /= float(len(decision_trees))
                neg_fraction /= float(len(decision_trees))

            pos_entropy = _calculate_entropy(np.divide(list(pos_avg_probs.values()), len(decision_trees)))
            neg_entropy = _calculate_entropy(np.divide(list(neg_avg_probs.values()), len(decision_trees)))

            pos_data = data[data[test[0]] <= test[1]].copy()
            neg_data = data[data[test[0]] > test[1]].copy()

            if not calc_fracs_from_ensemble:
                pos_fraction = float(len(pos_data)) / float(len(data))
                neg_fraction = float(len(neg_data)) / float(len(data))

            weighted_entropy = pos_fraction * pos_entropy + neg_fraction * neg_entropy
            information_gain = prior_entropy - weighted_entropy

            if information_gain > max_ig and len(pos_data) > 0 and len(neg_data) > 0:
                max_ig, best_dt.label, best_dt.value = information_gain, test[0], test[1]
                best_pos_data, best_neg_data, best_pos_entropy, best_neg_entropy = pos_data, neg_data, pos_entropy, neg_entropy

        # print max_ig
        if max_ig == 0:  # If we can't find a test that results in an information gain, we can pre-prune
            return decisiontree.DecisionTree(value=None, label=_get_most_occurring_class(data, class_label))

        # Update some variables and do recursive calls
        left_prior_tests = prior_tests.copy()
        left_prior_tests.update({(best_dt.label, best_dt.value): True})
        new_tests = tests.copy()
        new_tests.remove((best_dt.label, best_dt.value))
        best_dt.left = build_dt_from_ensemble(decision_trees, best_pos_data, class_label, new_tests,
                                              best_pos_entropy, left_prior_tests, min_nr_samples)

        right_prior_tests = prior_tests.copy()
        right_prior_tests.update({(best_dt.label, best_dt.value): False})
        best_dt.right = build_dt_from_ensemble(decision_trees, best_neg_data, class_label, new_tests,
                                               best_neg_entropy, right_prior_tests, min_nr_samples)

        return best_dt
    else:
        return decisiontree.DecisionTree(value=None, label=_get_most_occurring_class(data, class_label))