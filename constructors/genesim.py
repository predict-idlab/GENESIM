"""
Contains the code for the innovative algorithm called GENESIM

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

import copy
import multiprocessing
from collections import Counter

from pandas import DataFrame, concat

import numpy as np

import time
import sys

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from constructors.ensemble import bootstrap, RFClassification, XGBClassification
from decisiontree import DecisionTree

import os

class LineSegment(object):
    """
    Auxiliary class, used for the intersection algorithm. A region is composed of multiple lines (one for each
    dimension)
    """
    def __init__(self, lower_bound, upper_bound, region_index):
        self.lower_bound = lower_bound
        '''The lower bound of the line segment'''
        self.upper_bound = upper_bound
        '''The upper bound of the line segment'''
        self.region_index = region_index
        '''To which region does this line segment belong to'''


class GENESIM(object):

    def _decision_tree_to_decision_table(self, tree, feature_vectors):
        """
        Convert each path from the root to a leaf into a region, store it into a table
        :param tree: the constructed tree
        :param feature_vectors: the feature vectors of all samples
        :return: a set of regions in a k-dimensional space (k=|feature_vector|), corresponding to the decision tree
        """
        # Initialize an empty region (will be passed on recursively)
        region = {}
        for column in feature_vectors.columns:
            region[column] = [float("-inf"), float("inf")]
            region["class"] = None
        regions = self._tree_to_decision_table(tree, region, [])
        return regions

    def _tree_to_decision_table(self, tree, region, regions):
        """
        Recursive method used to convert the decision tree to a decision_table (do not call this one!)
        """
        left_region = copy.deepcopy(region)  # Take a deepcopy or we're fucked
        right_region = copy.deepcopy(region)  # Take a deepcopy or we're fucked
        left_region[tree.label][1] = tree.value
        right_region[tree.label][0] = tree.value

        if tree.left.value is None:
            if tree.left.class_probabilities is not None:
                left_region["class"] = tree.left.class_probabilities
            else:
                left_region["class"] = {tree.left.label: 1.0}
            regions.append(left_region)
        else:
            self._tree_to_decision_table(tree.left, left_region, regions)

        if tree.right.value is None:
            if tree.right.class_probabilities is not None:
                right_region["class"] = tree.right.class_probabilities
            else:
                left_region["class"] = {tree.left.label: 1.0}
            regions.append(right_region)
        else:
            self._tree_to_decision_table(tree.right, right_region, regions)

        return regions

    def _find_lines(self, regions, features, feature_mins, feature_maxs):
        if len(regions) <= 0: return {}

        for region in regions:
            for feature in features:
                if region[feature][0] == float("-inf"):
                    region[feature][0] = feature_mins[feature]
                if region[feature][1] == float("inf"):
                    region[feature][1] = feature_maxs[feature]

        lines = {}
        # First convert the region information into dataframes
        columns = []
        for feature in features:
            columns.append(feature+'_lb')
            columns.append(feature+'_ub')
        columns.append('class')
        regions_df = DataFrame(columns=columns)
        for region in regions:
            entry = []
            for feature in features:
                entry.append(region[feature][0])
                entry.append(region[feature][1])
            entry.append(region['class'])
            regions_df.loc[len(regions_df)] = entry

        for feature in features:
            other_features = list(set(features) - set([feature]))
            lb_bool_serie = [True]*len(regions_df)
            ub_bool_serie = [True]*len(regions_df)
            for other_feature in other_features:
                lb_bool_serie &= (regions_df[other_feature+'_lb'] == feature_mins[other_feature]).values
                ub_bool_serie &= (regions_df[other_feature+'_ub'] == feature_maxs[other_feature]).values

            lower_upper_regions = concat([regions_df[lb_bool_serie], regions_df[ub_bool_serie]])
            lines[feature] = []
            for value in np.unique(lower_upper_regions[lower_upper_regions.duplicated(feature+'_lb', False)][feature+'_lb']):
                if feature_mins[feature] != value and feature_maxs[feature] != value:
                    lines[feature].append(value)

        return lines

    def _regions_to_tree_improved(self, features_df, labels_df, regions, features, feature_mins, feature_maxs, max_samples=1):

        lines = self._find_lines(regions, features, feature_mins, feature_maxs)
        lines_keys = [key for key in lines.keys() if len(lines[key]) > 0]
        if lines is None or len(lines) <= 0 or len(lines_keys) <= 0:
            return DecisionTree(label=str(np.argmax(np.bincount(labels_df[labels_df.columns[0]].values.astype(int)))), value=None, data=features_df)

        random_label = np.random.choice(lines_keys)
        random_value = np.random.choice(lines[random_label])
        data = DataFrame(features_df)
        data[labels_df.columns[0]] = labels_df
        best_split_node = DecisionTree(data=data, label=random_label, value=random_value,
                            left=DecisionTree(data=data[data[random_label] <= random_value]),
                            right=DecisionTree(data=data[data[random_label] > random_value]))
        node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)

        feature_mins_right = feature_mins.copy()
        feature_mins_right[node.label] = node.value
        feature_maxs_left = feature_maxs.copy()
        feature_maxs_left[node.label] = node.value
        regions_left = []
        regions_right = []
        for region in regions:
            if region[best_split_node.label][0] < best_split_node.value:
                regions_left.append(region)
            else:
                regions_right.append(region)
        if len(best_split_node.left.data) >= max_samples and len(best_split_node.right.data) >= max_samples:
            node.left = self._regions_to_tree_improved(best_split_node.left.data.drop(labels_df.columns[0], axis=1),
                                                       best_split_node.left.data[[labels_df.columns[0]]], regions_left, features,
                                                       feature_mins, feature_maxs_left)
            node.right = self._regions_to_tree_improved(best_split_node.right.data.drop(labels_df.columns[0], axis=1),
                                                        best_split_node.right.data[[labels_df.columns[0]]], regions_right, features,
                                                        feature_mins_right, feature_maxs)

        else:
            node.label = str(np.argmax(np.bincount(labels_df[labels_df.columns[0]].values.astype(int))))
            node.value = None

        return node

    def _intersect(self, line1_lb, line1_ub, line2_lb, line2_ub):
        if line1_ub <= line2_lb: return False
        if line1_lb >= line2_ub: return False
        return True

    def _calculate_intersection(self, regions1, regions2, features, feature_maxs, feature_mins):
        """
        Fancy method to calculate intersections. O(d*n*log(n)) instead of O(d*n^2)

        Instead of brute force, we iterate over each possible dimension,
        we project each region to that one dimension, creating a line segment. We then construct a set S_i for each
        dimension containing pairs of line segments that intersect in dimension i. In the end, the intersection
        of all these sets results in the intersecting regions. For all these intersection regions, their intersecting
        region is calculated and added to a new set, which is returned in the end
        :param regions1: first set of regions
        :param regions2: second set of regions
        :param features: list of dimension names
        :return: new set of regions, which are the intersections of the regions in 1 and 2
        """
        # print "Merging ", len(regions1), " with ", len(regions2), " regions."
        S_intersections = [None] * len(features)
        for i in range(len(features)):
            # Create B1 and B2: 2 arrays of line segments
            box_set1 = []
            for region_index in range(len(regions1)):
                box_set1.append(LineSegment(regions1[region_index][features[i]][0], regions1[region_index][features[i]][1],
                                            region_index))
            box_set2 = []
            for region_index in range(len(regions2)):
                box_set2.append(LineSegment(regions2[region_index][features[i]][0], regions2[region_index][features[i]][1],
                                            region_index))

            # Sort the two boxsets by their lower bound
            box_set1 = sorted(box_set1, key=lambda segment: segment.lower_bound)
            box_set2 = sorted(box_set2, key=lambda segment: segment.lower_bound)

            # Create a list of unique lower bounds, we iterate over these bounds later
            unique_lower_bounds = []
            for j in range(max(len(box_set1), len(box_set2))):
                if j < len(box_set1) and box_set1[j].lower_bound not in unique_lower_bounds:
                    unique_lower_bounds.append(box_set1[j].lower_bound)

                if j < len(box_set2) and box_set2[j].lower_bound not in unique_lower_bounds:
                    unique_lower_bounds.append(box_set2[j].lower_bound)

            # Sort them
            unique_lower_bounds = sorted(unique_lower_bounds)

            box1_active_set = []
            box2_active_set = []
            intersections = []
            for lower_bound in unique_lower_bounds:
                # Update all active sets, a region is added when it's lower bound is lower than the current one
                # It is removed when its upper bound is higher than the current lower bound
                for j in range(len(box_set1)):
                    if box_set1[j].upper_bound <= lower_bound:
                        if box_set1[j] in box1_active_set:
                            box1_active_set.remove(box_set1[j])
                    elif box_set1[j].lower_bound <= lower_bound:
                        if box_set1[j] not in box1_active_set:
                            box1_active_set.append(box_set1[j])
                    else:
                        break

                for j in range(len(box_set2)):
                    if box_set2[j].upper_bound <= lower_bound:
                        if box_set2[j] in box2_active_set:
                            box2_active_set.remove(box_set2[j])
                    elif box_set2[j].lower_bound <= lower_bound:
                        if box_set2[j] not in box2_active_set:
                            box2_active_set.append(box_set2[j])
                    else:
                        break

                # All regions from the active set of B1 intersect with the regions in the active set of B2
                for segment1 in box1_active_set:
                    for segment2 in box2_active_set:
                        intersections.append((segment1.region_index, segment2.region_index))

            S_intersections[i] = intersections

        # The intersection of all these S_i's are the intersecting regions
        intersection_regions_indices = S_intersections[0]
        for k in range(1, len(S_intersections)):
            intersection_regions_indices = self._tuple_list_intersections(intersection_regions_indices, S_intersections[k])

        # Create a new set of regions
        intersected_regions = []
        for intersection_region_pair in intersection_regions_indices:
            region = {}
            for feature in features:
                region[feature] = [max(regions1[intersection_region_pair[0]][feature][0],
                                       regions2[intersection_region_pair[1]][feature][0]),
                                   min(regions1[intersection_region_pair[0]][feature][1],
                                       regions2[intersection_region_pair[1]][feature][1])]
                # Convert all -inf and inf to the mins and max from those features
                if region[feature][0] == float("-inf"):
                    region[feature][0] = feature_mins[feature]
                if region[feature][1] == float("inf"):
                    region[feature][1] = feature_maxs[feature]
            region['class'] = {}
            for key in set(set(regions1[intersection_region_pair[0]]['class'].iterkeys()) |
                                   set(regions2[intersection_region_pair[1]]['class'].iterkeys())):
                prob_1 = (regions1[intersection_region_pair[0]]['class'][key]
                          if key in regions1[intersection_region_pair[0]]['class'] else 0)
                prob_2 = (regions2[intersection_region_pair[1]]['class'][key]
                          if key in regions2[intersection_region_pair[1]]['class'] else 0)
                if prob_1 and prob_2:
                    region['class'][key] = (regions1[intersection_region_pair[0]]['class'][key] +
                                            regions2[intersection_region_pair[1]]['class'][key]) / 2
                else:
                    if prob_1:
                        region['class'][key] = prob_1
                    else:
                        region['class'][key] = prob_2
            intersected_regions.append(region)

        return intersected_regions

    def _tuple_list_intersections(self, list1, list2):
        # Make sure the length of list1 is larger than the length of list2
        if len(list2) > len(list1):
            return self._tuple_list_intersections(list2, list1)
        else:
            list1 = set(list1)
            list2 = set(list2)
            intersections = []
            for tuple in list2:
                if tuple in list1:
                    intersections.append(tuple)

            return intersections

    def _fitness(self, tree, test_features_df, test_labels_df, cat_name, alpha=1, beta=0):
        return alpha*(1-accuracy_score(test_labels_df[cat_name].values.astype(int),
                                       tree.evaluate_multiple(test_features_df).astype(int))) + beta*tree.count_nodes()

    def _mutate_shift_random(self, tree, feature_vectors, labels):
        # tree.visualise('beforeMutationShift')
        internal_nodes = list(set(tree._get_nodes()) - set(tree._get_leaves()))
        tree = copy.deepcopy(tree)
        # print 'nr internal nodes =', len(internal_nodes)
        if len(internal_nodes) > 1:
            random_node = np.random.choice(internal_nodes)
            random_value = np.random.choice(np.unique(feature_vectors[random_node.label].values))
            random_node.value = random_value
            tree.populate_samples(feature_vectors, labels)
        # tree.visualise('afterMutationShift')
        # raw_input()
        return tree

    def _mutate_swap_subtrees(self, tree, feature_vectors, labels):
        # tree.visualise('beforeMutationSwap')
        tree = copy.deepcopy(tree)
        tree._set_parents()
        nodes = tree._get_nodes()
        if len(nodes) > 1:
            node1 = np.random.choice(nodes)
            node2 = np.random.choice(nodes)
            parent1 = node1.parent
            parent2 = node2.parent

            if parent1 is not None and parent2 is not None: # We don't want the root node
                if parent1.left == node1:
                    parent1.left = node2
                else:
                    parent1.right = node2

                if parent2.left == node2:
                    parent2.left = node1
                else:
                    parent2.right = node1

                tree.populate_samples(feature_vectors, labels)

        return tree

    def _tournament_selection_and_merging(self, trees, train_features_df, train_labels_df, test_features_df,
                                          test_labels_df, cat_name, feature_cols, feature_maxs, feature_mins,
                                          max_samples, return_dict, seed, tournament_size=3):
        np.random.seed(seed)
        _tournament_size = min(len(trees) / 2, tournament_size)
        trees = copy.deepcopy(trees)
        best_fitness_1 = sys.float_info.max
        best_tree_1 = None
        for i in range(_tournament_size):
            tree = np.random.choice(trees)
            trees.remove(tree)
            fitness = self._fitness(tree, test_features_df, test_labels_df, cat_name)
            if tree is not None and tree.count_nodes() > 1 and fitness < best_fitness_1:
                best_fitness_1 = fitness
                best_tree_1 = tree
        best_fitness_2 = sys.float_info.max
        best_tree_2 = None
        for i in range(_tournament_size):
            tree = np.random.choice(trees)
            trees.remove(tree)
            fitness = self._fitness(tree, test_features_df, test_labels_df, cat_name)
            if tree is not None and tree.count_nodes() > 1 and fitness < best_fitness_2:
                best_fitness_2 = fitness
                best_tree_2 = tree

        if best_tree_1 is not None and best_tree_2 is not None:
            region1 = self._decision_tree_to_decision_table(best_tree_1, train_features_df)
            region2 = self._decision_tree_to_decision_table(best_tree_2, train_features_df)
            merged_regions = self._calculate_intersection(region1, region2, feature_cols, feature_maxs, feature_mins)
            return_dict[seed] = self._regions_to_tree_improved(train_features_df, train_labels_df, merged_regions, feature_cols,
                                                               feature_mins, feature_maxs, max_samples=max_samples)
            return return_dict[seed]
        else:
            return_dict[seed] = None
            return None

    def _convert_sklearn_to_tree(self, dt, features):
        """Convert a sklearn object to a `decisiontree.decisiontree` object"""
        n_nodes = dt.tree_.node_count
        children_left = dt.tree_.children_left
        children_right = dt.tree_.children_right
        feature = dt.tree_.feature
        threshold = dt.tree_.threshold
        classes = dt.classes_

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes)
        decision_trees = [None] * n_nodes
        for i in range(n_nodes):
            decision_trees[i] = DecisionTree()
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        for i in range(n_nodes):

            if children_left[i] > 0:
                decision_trees[i].left = decision_trees[children_left[i]]

            if children_right[i] > 0:
                decision_trees[i].right = decision_trees[children_right[i]]

            if is_leaves[i]:
                decision_trees[i].label = dt.classes_[np.argmax(dt.tree_.value[i][0])]
                decision_trees[i].value = None
            else:
                decision_trees[i].label = features[feature[i]]
                decision_trees[i].value = threshold[i]

        return decision_trees[0]

    def parse_xgb_tree_string(self, tree_string, training_data, feature_cols, label_col, the_class):
        # Get class distribution
        _classes = np.unique(training_data[label_col].values)
        class_distributions = {}
        for _class in _classes:
            data = training_data[training_data[label_col] != _class]
            class_counts = Counter(data[label_col].values)
            total = sum(class_counts.values(), 0.0)
            for key in class_counts:
                class_counts[key] /= total
            class_distributions[_class] = class_counts

        # Get the unique values per feature
        unique_values_per_feature = {}
        for feature_col in feature_cols:
            # Just use a simple sorted list of np.unique (faster than SortedList or a set to get index of elements after testing)
            unique_values_per_feature[feature_col] = sorted(np.unique(training_data[feature_col].values))

        return self.parse_xgb_tree(tree_string, _class=the_class, class_distributions=class_distributions,
                                   unique_values_per_feature=unique_values_per_feature, n_samples=len(training_data))

    def get_closest_value(self, x, values):
        for i in range(len(values) - 1):
            if values[i + 1] > x:
                return float(values[i])
        return x

    def parse_xgb_tree(self, tree_string, _class=0, class_distributions={}, unique_values_per_feature={}, n_samples=0):
        # There is some magic involved! The leaf values need to be converted to class distributions somehow!
        # For binary classification problems: convert to probability by calculating 1/(1+exp(-value))
        # For multi_class: the tree_string contains n_estimators * n_classes decision trees
        # WARNING: Classes are sorted according to output of np.unique
        # Ordered as follows [tree_1-class_1, ..., tree_1-class_k, tree_2-class_1, ....]

        # The problem is: tree_i is different for each class...
        # One possibility is to assign the probability to that class by calculating logistic function
        # And dividing the rest of the probability (sum to 1) according to the distribution of the remaining classes

        # Next problem is: everything is expressed as "feature < threshold" instead "feature <= threshold"
        # Solution is: take the infimum of those feature values

        decision_trees = {}
        # Binary classification
        binary_classification = len(class_distributions.keys()) == 2
        for line in tree_string.split('\n'):
            if line != '':
                _id, rest = line.split(':')
                _id = _id.lstrip()
                if rest[:4] != 'leaf':
                    feature = rest.split('<')[0][1:]
                    highest_lower_threshold = self.get_closest_value(float(rest.split('<')[1].split(']')[0]),
                                                                unique_values_per_feature[feature])
                    decision_trees[_id] = DecisionTree(right=None, left=None,
                                                       label=feature, value=highest_lower_threshold,
                                                       parent=None)
                else:
                    leaf_value = float(rest.split('=')[1])
                    if binary_classification:
                        probability = 1 / (1 + np.exp(leaf_value))
                        other_class = class_distributions[_class].keys()[0]
                        class_probs = {_class: int(n_samples * probability),
                                       other_class: int(n_samples * (1 - probability))}
                        if probability > 0.5:
                            most_probable_class = _class
                        else:
                            most_probable_class = other_class
                    else:
                        probability = 1 / (1 + np.exp(-leaf_value))
                        class_probs = {}
                        remainder_samples = int(n_samples - probability * n_samples)
                        class_probs[_class] = int(probability * n_samples)
                        most_probable_class, most_samples = _class, class_probs[_class]
                        for other_class in class_distributions[_class]:
                            amount_samples = int(remainder_samples * class_distributions[_class][other_class])
                            class_probs[other_class] = amount_samples
                            if amount_samples > most_samples:
                                most_probable_class = other_class
                                most_samples = amount_samples

                    decision_trees[_id] = DecisionTree(right=None, left=None,
                                                       label=most_probable_class, value=None,
                                                       parent=None)
                    decision_trees[_id].class_probabilities = class_probs

        # Make another pass to link the different decision trees together
        for line in tree_string.split('\n'):
            if line != '':
                _id, rest = line.split(':')
                _id = _id.lstrip()
                tree = decision_trees[_id]
                if rest[:4] != 'leaf':
                    rest = rest.split(']')[1].lstrip()
                    links = rest.split(',')
                    for link in links:
                        word, link_id = link.split('=')
                        if word == 'yes' or word == 'missing':
                            tree.left = decision_trees[link_id]
                        else:
                            tree.right = decision_trees[link_id]

        return decision_trees['0']

    def genetic_algorithm(self, data, label_col, tree_constructors, population_size=15, num_crossovers=3, val_fraction=0.25,
                          num_iterations=5, seed=1337, tournament_size=3, prune=False, max_samples=3,
                          nr_bootstraps=5, mutation_prob=0.25, tree_path='trees'):
        """
        Construct an ensemble using different induction algorithms, combined with bagging and bootstrapping and convert
        it to a single, interpretable model

        **Params**
        ----------
          - `data` (pandas DataFrame) - a `Dataframe` containing all the training data

          - `label_col` (string) - the name of the class label column

          - `tree_constructors` (list) - which induction algorithms must be used to create an
          ensemble with. Must be of type `constructors.treeconstructor.TreeConstructor`

          - `population_size` (int) - the maximum size of the population, the least fittest ones get discarded

          - `num_crossovers` (int) - how many pairs of decision trees get merged in each iteration

          - `val_fraction` (int) - how much percent of the data will be used as validation set to calculate fitness with

          - `num_iterations` (int) - how many iterations does the algorithm need to run

          - `seed` (int) - a seed for reproducible results

          - `prune` (boolean) - if `True`, then prune each obtained tree by using the validation set

          - `max_samples` (int) - pre-prune condition when converting a region back to a tree

          - `nr_bootstraps` (int) - how many bootstraps are used to created the ensemble with?

          - `mutation_prob` (float) - how much percent chance does an individual have to mutate each iteration

        **Returns**
        -----------
            a DecisionTree object
        """
	if not os.path.exists(tree_path):
	    os.makedirs(tree_path)
        np.random.seed(seed)

        feature_mins = {}
        feature_maxs = {}
        feature_column_names = list(set(data.columns) - set([label_col]))

        for feature in feature_column_names:
            feature_mins[feature] = np.min(data[feature])
            feature_maxs[feature] = np.max(data[feature])

        labels_df = DataFrame()
        labels_df[label_col] = data[label_col].copy()
        features_df = data.copy()
        features_df = features_df.drop(label_col, axis=1)

        data = features_df.copy()
        data[label_col] = labels_df[label_col]

        sss = StratifiedShuffleSplit(labels_df[label_col], 1, test_size=val_fraction, random_state=seed)

        for train_index, test_index in sss:
            train_features_df, test_features_df = features_df.iloc[train_index, :].copy(), features_df.iloc[test_index,
                                                                                           :].copy()
            train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index,
                                                                                     :].copy()
            train_features_df = train_features_df.reset_index(drop=True)
            test_features_df = test_features_df.reset_index(drop=True)
            train_labels_df = train_labels_df.reset_index(drop=True)
            test_labels_df = test_labels_df.reset_index(drop=True)
            train = data.iloc[train_index, :].copy().reset_index(drop=True)

        tree_list = bootstrap(train, label_col, tree_constructors, boosting=True, nr_classifiers=nr_bootstraps)
        for constructor in tree_constructors:
            tree = constructor.construct_classifier(train, train_features_df.columns, label_col)
            tree.populate_samples(train_features_df, train_labels_df[label_col].values)
            tree_list.append(tree)


        # Adding the random forest trees to the population
        rf = RFClassification()
        xgb = XGBClassification()

        feature_cols = list(train_features_df.columns)
        rf.construct_classifier(train, feature_cols, label_col)
        xgb_model = xgb.construct_classifier(train, feature_cols, label_col)

        # print 'Random forest number of estimators:', len(rf.clf.estimators_)

        for i, estimator in enumerate(rf.clf.estimators_):
            tree = self._convert_sklearn_to_tree(estimator, feature_cols)
            # print tree.get_binary_vector(train_features_df.iloc[0])
            tree.populate_samples(train_features_df, train_labels_df[label_col].values)
            predicted_labels = tree.evaluate_multiple(test_features_df).astype(int)
            # accuracy = accuracy_score(test_labels_df[label_col].values.astype(str), predicted_labels.astype(str))
            # print 'RF tree', i, '/', len(rf.clf.estimators_), ':', accuracy
            tree_list.append(tree)

        n_classes = len(np.unique(train[label_col].values))
        if n_classes > 2:
            for idx, tree_string in enumerate(xgb_model.clf._Booster.get_dump()):
                tree = self.parse_xgb_tree_string(tree_string, train, feature_cols, label_col,
                                             np.unique(train[label_col].values)[idx % n_classes])
                tree_list.append(tree)
        else:
            for tree_string in xgb_model.clf._Booster.get_dump():
                tree = self.parse_xgb_tree_string(tree_string, train, feature_cols, label_col, 0)
                tree_list.append(tree)

        tree_list = [tree for tree in tree_list if tree is not None ]

        start = time.clock()

        for k in range(num_iterations):
            # print "Calculating accuracy and sorting"
            tree_accuracy = []
            for tree in tree_list:
                predicted_labels = tree.evaluate_multiple(test_features_df)
                accuracy = accuracy_score(test_labels_df[label_col].values.astype(int), predicted_labels.astype(int))
                tree_accuracy.append((tree, accuracy, tree.count_nodes()))

            tree_list = [x[0] for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]]
            best_tree = tree_list[0]
            with open(tree_path+os.sep+'it_'+str(k)+'.tree', 'w+') as fp:
                fp.write(best_tree.convert_to_json())

            # print("----> Best tree till now: ", [(x[1], x[2]) for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]])

            # Crossovers
            #mngr = multiprocessing.Manager()
            #return_dict = mngr.dict()
            #jobs = []
            for i in range(num_crossovers):
               # p = multiprocessing.Process(target=self._tournament_selection_and_merging, args=[tree_list, train_features_df, train_labels_df,
               #                                                                                  test_features_df, test_labels_df, label_col,
               #                                                                                  feature_column_names, feature_maxs, feature_mins,
               #                                                                                  max_samples, return_dict, k * i + i, tournament_size])
               # jobs.append(p)
               # p.start()


            # for proc in jobs:
            #     proc.join()
		new_tree = self._tournament_selection_and_merging(tree_list, train_features_df, train_labels_df, test_features_df, test_labels_df, label_col,
								  feature_column_names, feature_maxs, feature_mins, max_samples, {}, k*i+i, tournament_size) 
            # for new_tree in return_dict.values():
                if new_tree is not None:
                    # print 'new tree added', accuracy_score(test_labels_df[label_col].values.astype(int), new_tree.evaluate_multiple(test_features_df).astype(int))
                    tree_list.append(new_tree)

                    if prune:
                        # print 'Pruning the tree...', new_tree.count_nodes()
                        new_tree = new_tree.cost_complexity_pruning(train_features_df, train_labels_df[label_col], None, cv=False,
                                                            val_features=test_features_df,
                                                            val_labels=test_labels_df[label_col])
                        # print 'Done', new_tree.count_nodes(), accuracy_score(test_labels_df[label_col].values.astype(int), new_tree.evaluate_multiple(test_features_df).astype(int))
                        tree_list.append(new_tree)

            # Mutation phase
            for tree in tree_list:
                value = np.random.rand()
                if value < mutation_prob:
                    new_tree1 = self._mutate_shift_random(tree, train_features_df, train_labels_df[label_col].values)
                    # print 'new mutation added', accuracy_score(test_labels_df[label_col].values.astype(int),
                                                               #new_tree1.evaluate_multiple(test_features_df).astype(int))
                    new_tree2 = self._mutate_swap_subtrees(tree, train_features_df, train_labels_df[label_col].values)
                    # print 'new mutation added', accuracy_score(test_labels_df[label_col].values.astype(int),
                                                               #new_tree2.evaluate_multiple(test_features_df).astype(int))
                    tree_list.append(new_tree1)
                    tree_list.append(new_tree2)


            end = time.clock()
            print "Took ", (end - start), " seconds"
            start = end

        tree_accuracy = []
        for tree in tree_list:
            predicted_labels = tree.evaluate_multiple(test_features_df)
            accuracy = accuracy_score(test_labels_df[label_col].values.astype(int), predicted_labels.astype(int))
            tree_accuracy.append((tree, accuracy, tree.count_nodes()))


        # print [x for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]]

        best_tree = sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[0][0]
        return best_tree
