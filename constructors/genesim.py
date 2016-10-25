"""
Contains the code for the innovative algorithm called GENESIM

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

import copy
import multiprocessing
from pandas import DataFrame, concat

import numpy as np

import time
import sys

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from constructors.ensemble import bootstrap
from decisiontree import DecisionTree

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
        print "Merging ", len(regions1), " with ", len(regions2), " regions."
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
        return alpha*(1-accuracy_score(test_labels_df[cat_name].values.astype(str),
                                       tree.evaluate_multiple(test_features_df).astype(str))) + beta*tree.count_nodes()

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
            # return 0
        else:
            return_dict[seed] = None
            # return 0

    def genetic_algorithm(self, data, label_col, tree_constructors, population_size=15, num_crossovers=3, val_fraction=0.25,
                          num_iterations=5, seed=1337, tournament_size=3, prune=False, max_samples=3,
                          nr_bootstraps=5, mutation_prob=0.25):
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
        tree_list = [tree for tree in tree_list if tree is not None ]

        start = time.clock()

        for k in range(num_iterations):
            print "Calculating accuracy and sorting"
            tree_accuracy = []
            for tree in tree_list:
                predicted_labels = tree.evaluate_multiple(test_features_df)
                accuracy = accuracy_score(test_labels_df[label_col].values.astype(str), predicted_labels.astype(str))
                tree_accuracy.append((tree, accuracy, tree.count_nodes()))

            tree_list = [x[0] for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]]
            print("----> Best tree till now: ", [(x[1], x[2]) for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]])

            # Crossovers
            mngr = multiprocessing.Manager()
            return_dict = mngr.dict()
            jobs = []
            for i in range(num_crossovers):
                p = multiprocessing.Process(target=self._tournament_selection_and_merging, args=[tree_list, train_features_df, train_labels_df,
                                                                                                 test_features_df, test_labels_df, label_col,
                                                                                                 feature_column_names, feature_maxs, feature_mins,
                                                                                                 max_samples, return_dict, k * i + i, tournament_size])
                jobs.append(p)
                p.start()


            for proc in jobs:
                proc.join()

            for new_tree in return_dict.values():
                if new_tree is not None:
                    print 'new tree added', accuracy_score(test_labels_df[label_col].values.astype(str), new_tree.evaluate_multiple(test_features_df).astype(str))
                    tree_list.append(new_tree)

                    if prune:
                        print 'Pruning the tree...', new_tree.count_nodes()
                        new_tree = new_tree.cost_complexity_pruning(train_features_df, train_labels_df[label_col], None, cv=False,
                                                            val_features=test_features_df,
                                                            val_labels=test_labels_df[label_col])
                        print 'Done', new_tree.count_nodes(), accuracy_score(test_labels_df[label_col].values.astype(str), new_tree.evaluate_multiple(test_features_df).astype(str))
                        tree_list.append(new_tree)

            # Mutation phase
            for tree in tree_list:
                value = np.random.rand()
                if value < mutation_prob:
                    new_tree1 = self._mutate_shift_random(tree, train_features_df, train_labels_df[label_col].values)
                    print 'new mutation added', accuracy_score(test_labels_df[label_col].values.astype(str),
                                                               new_tree1.evaluate_multiple(test_features_df).astype(str))
                    new_tree2 = self._mutate_swap_subtrees(tree, train_features_df, train_labels_df[label_col].values)
                    print 'new mutation added', accuracy_score(test_labels_df[label_col].values.astype(str),
                                                               new_tree2.evaluate_multiple(test_features_df).astype(str))
                    tree_list.append(new_tree1)
                    tree_list.append(new_tree2)


            end = time.clock()
            print "Took ", (end - start), " seconds"
            start = end

        tree_accuracy = []
        for tree in tree_list:
            predicted_labels = tree.evaluate_multiple(test_features_df)
            accuracy = accuracy_score(test_labels_df[label_col].values.astype(str), predicted_labels.astype(str))
            tree_accuracy.append((tree, accuracy, tree.count_nodes()))


        print [x for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]]

        best_tree = sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[0][0]
        return best_tree