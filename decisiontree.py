"""
Contains the decisiontree object used throughout this project.

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""
from copy import deepcopy, copy

import sklearn
from graphviz import Source
import matplotlib.pyplot as plt
import numpy as np
import json
import operator

from pandas import Series
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

import constructors.ISM
import constructors.ensemble


class DecisionTree(object):
    """This class contains the main object used throughout this project: a decision tree. It contains methods
    to visualise and evaluate the trees."""

    def __init__(self, right=None, left=None, label='', value=None, data=None, parent=None):
        """Create a node of a decision tree"""
        self.right = right
        '''right child, taken when a sample[`decisiontree.DecisionTree.label`] > `decisiontree.DecisionTree.value`'''
        self.left = left
        '''left child, taken when sample[`decisiontree.DecisionTree.label`] <= `decisiontree.DecisionTree.value`'''
        self.label = label
        '''string representation of the attribute the node splits on'''
        self.value = value
        '''the value where the node splits on (if `None`, then we're in a leaf)'''
        self.data = data
        '''dataframe of samples in the subtree'''
        self.parent = parent
        '''the parent of the node (used for pruning)'''
        self.class_probabilities = {}
        '''probability estimates for the leaves'''

    def evaluate(self, feature_vector):
        """Create a prediction for a sample (using its feature vector)

        **Params**
        ----------
          - `feature_vector` (pandas Series or dict) - the sample to evaluate, must be a `pandas Series` object or a
          `dict`. It is important that the attribute keys in the sample are the same as the labels occuring in the tree.

        **Returns**
        -----------
            the predicted class label
        """
        if self.value is None:
            return self.label
        else:
            # feature_vector should only contain 1 row
            if feature_vector[self.label] <= self.value:
                return self.left.evaluate(feature_vector)
            else:
                return self.right.evaluate(feature_vector)

    def evaluate_multiple(self, feature_vectors):
        """Create a prediction for multiple samples by calling `decisiontree.DecisionTree.evaluate`

        **Params**
        ----------
          - `feature_vectors` (pandas DataFrame or list of dict) - the samples to evaluate

        **Returns**
        -----------
            a list of predicted class labels for each sample
        """
        results = []

        for _index, feature_vector in feature_vectors.iterrows():
            results.append(self.evaluate(feature_vector))
        return np.asarray(results)

    def plot_confusion_matrix(self, actual_labels, feature_vectors, normalized=False, plot=False):
        """Create a prediction for each of the samples and compare them to the actual labels by generating
        a confusion matrix

        **Params**
        ----------
          - `actual_labels` (pandas Series or list) - the real class labels

          - `feature_vectors` (pandas DataFrame or list of dict) - the samples to evaluate

          - `normalized` (boolean) - if `True`, then generate a normalized confusion matrix

          - `plot` (boolean) - if `True`, then generate a plot.

        **Returns**
        -----------
            the confusion matrix
        """
        predicted_labels = self.evaluate_multiple(feature_vectors)
        confusion_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels)
        if plot:
            confusion_matrix.plot(normalized=normalized)
            plt.show()

        return confusion_matrix

    def cost_complexity_pruning(self, feature_vectors, labels, tree_constructor, n_folds=3, cv=True, val_features=None,
                                val_labels=None, ism_constructors=[], ism_calc_fracs=False, ism_nr_classifiers=3,
                                ism_boosting=False):
        """Apply cost-complexity pruning ([\[1\]](http://mlwiki.org/index.php/Cost-Complexity_Pruning),
        [\[2\]](http://support.sas.com/documentation/cdl/en/stathpug/68163/HTML/default/viewer.htm#stathpug_hpsplit_details06.htm),
        [\[3\]](ftp://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/TREE-pruning.pdf))
        to the tree

        **Params**
        ----------
          - `feature_vectors` (pandas DataFrame or list of dict) - the training feature vectors for each sample

          - `labels` (pandas Series or list) - the training class labels for each sample

          - `tree_constructor` (boolean) - with which `constructors.treeconstructor.TreeConstructor` is the original decision tree made

          - `n_folds` (int) - if `cv` is `True`, this is the number of folds used to calculate the best alpha

          - `cv` (boolean) - if this is `True`, cross-validation will be applied to calculate the best alpha,
          else a validation set is used

          - `val_features` (pandas DataFrame or list of dict) - if `cv` is `False`, these are the validation feature vectors to calculate best alpha

          - `val_labels` (pandas Series or list) - if `cv` is `False`, these are the validation class labels to calculate best alpha

          - `ism_constructors` (list) - a list of constructors from which to construct the ensemble

          - `ism_calc_fracs` (boolean) - if `True`, then all probabilities are estimated by using the ensemble

          - `ism_nr_classifiers` (int) - how many times do we need to bag for each of the constructors?

          - `ism_boosting` (boolean) - only used when the model to prune is ISM. When this is `True`, boosting will be applied too to create an ensemble

        **Returns**
        -----------
            a pruned decision tree
        """
        self._set_parents()
        self.populate_samples(feature_vectors, labels.values)
        root_samples = sum(self.class_probabilities.values())

        betas = []
        subtrees = self._generate_subtree_sequence(root_samples)
        if len(subtrees) == 0: return self  # Something went wrong, pruning failed

        if not cv:
            __min = (1, 99)
            best_tree = None
            for tree in subtrees:
                predictions = tree.evaluate_multiple(val_features).astype(int)
                err, nodes = 1 - accuracy_score(val_labels.values, predictions), tree.count_nodes()
                if (err, nodes) < __min:
                    __min = (err, nodes)
                    best_tree = tree
            return best_tree
        else:
            subtrees_by_alpha = {y:x for x,y in subtrees.iteritems()}
            subtrees_by_beta = {}
            alphas = sorted(subtrees.values())
            for i in range(len(alphas)-1):
                beta = np.sqrt(alphas[i]*alphas[i+1])
                betas.append(beta)
                subtrees_by_beta[beta] = subtrees_by_alpha[alphas[i]]
            beta_errors = {}
            for beta in betas:
                beta_errors[beta] = []

            skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
            for train_index, test_index in skf:
                X_train = feature_vectors.iloc[train_index, :].reset_index(drop=True)
                y_train = labels.iloc[train_index].reset_index(drop=True)
                train = X_train.copy()
                train[y_train.name] = Series(y_train, index=train.index)
                X_test = feature_vectors.iloc[test_index, :].reset_index(drop=True)
                y_test = labels.iloc[test_index].reset_index(drop=True)
                if tree_constructor == 'ism':
                    constructed_tree = constructors.ISM.ism(constructors.ensemble.bootstrap(train, y_train.name, ism_constructors,
                                                         nr_classifiers=ism_nr_classifiers, boosting=ism_boosting),
                                               train, y_train.name, calc_fracs_from_ensemble=ism_calc_fracs)
                else:
                    constructed_tree = tree_constructor.construct_classifier(train, X_train.columns, y_train.name)
                for beta in betas:
                    tree_copy = deepcopy(constructed_tree)
                    tree_copy.populate_samples(X_train, y_train.values)
                    pruned_tree = tree_copy._minimize_cost_complexity(root_samples, beta)
                    predictions = pruned_tree.evaluate_multiple(X_test).astype(int)
                    beta_errors[beta].append(1 - accuracy_score(predictions, y_test))

            for beta in beta_errors: beta_errors[beta] = np.mean(beta_errors[beta])
            return subtrees_by_beta[min(beta_errors.iteritems(), key=operator.itemgetter(1))[0]]


    # <editor-fold desc="Visualisation and conversion methods">

    def visualise(self, output_path, _view=True, show_probabilities=True):
        """Visualise the tree with [graphviz](http://www.graphviz.org/),
         using `decisiontree.DecisionTree.convert_to_dot`

        **Params**
        ----------
          - `output_path` (string) - where the file needs to be saved

          - `show_probabilities` (boolean) - if this is `True`, probabilities will be displayed in the leafs too

          - `_view` (boolean) - open the pdf after generation or not

        **Returns**
        -----------
            a pdf with the rendered dot code of the tree
        """
        src = Source(self.convert_to_dot(show_probabilities=show_probabilities))
        src.render(output_path, view=_view)

    def _get_number_of_subnodes(self, count=0):
        """Private method using in convert_node_to_dot, in order to give the right child of a node the right count

        :param count: intern parameter, don't set it
        :return: the number of subnodes of a specific node, not including himself
        """
        if self.value is None:
            return count
        else:
            return self.left._get_number_of_subnodes(count=count + 1) + self.right._get_number_of_subnodes(count=count + 1)

    def convert_to_dot(self, show_probabilities=True):
        """Converts a decision tree object to DOT code

        **Params**
        ----------
          - `show_probabilities` (boolean) - if this is `True`, probabilities will be displayed in the leafs too

        **Returns**
        -----------
            a string with the dot code for the decision tree
        """
        s = 'digraph DT{\n'
        s += 'node[fontname="Arial"];\n'
        s += self._convert_node_to_dot(show_probabilities=show_probabilities)
        s += '}'
        return s

    def _convert_node_to_dot(self, count=1, show_probabilities=True):
        """Convert node to dot format in order to visualize our tree using graphviz

        :param count: parameter used to give nodes unique names\
        :param show_probabilities: if this is True, probabilities will be plotted in the leafs too
        :return: intermediate string of the tree in dot format, without preamble (this is no correct dot format yet!)
        """
        if self.value is None:
            if len(self.class_probabilities) > 0 and show_probabilities:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + '\n'+self.class_probabilities.__str__()+'" shape="box"];\n'
            else:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + '" shape="box"];\n'
        else:
            if len(self.class_probabilities) > 0 and show_probabilities:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + ' <= ' + str(self.value) + '\n'+self.class_probabilities.__str__() +'"];\n'
            else:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + ' <= ' + str(self.value) + '"];\n'
            s += self.left._convert_node_to_dot(count=count + 1)
            s += 'Node' + str(count) + ' -> ' + 'Node' + str(count + 1) + ' [label="true"];\n'
            number_of_subnodes = self.left._get_number_of_subnodes()
            s += self.right._convert_node_to_dot(count=count + number_of_subnodes + 2)
            s += 'Node' + str(count) + '->' + 'Node' + str(count + number_of_subnodes + 2) + ' [label="false"];\n'

        return s

    def convert_to_string(self, tab=0):
        """Converts a decision tree object to a string representation

        **Params**
        ----------
          - `tab` (int) - recursive parameter to tabulate the different levels of the tree

        **Returns**
        -----------
            a string representation of the decision tree
        """
        if self.value is None:
            print('\t' * tab + '[', self.label, ']')
        else:
            print('\t' * tab + self.label, ' <= ', str(self.value))
            print('\t' * (tab + 1) + 'LEFT:')
            self.left.convert_to_string(tab=tab + 1)
            print('\t' * (tab + 1) + 'RIGHT:')
            self.right.convert_to_string(tab=tab + 1)

    def convert_to_json(self):
        """Converts a decision tree object to a JSON representation

        **Returns**
        -----------
            a string with the JSON code for the decision tree
        """
        json = "{\n"
        json += "\t\"name\": \"" + str(self.label) + " <= " + str(self.value) + "\",\n"
        json += "\t\"rule\": \"null\",\n"
        json += "\t\"children\": [\n"
        json += DecisionTree._convert_node_to_json(self.left, "True") + ",\n"
        json += DecisionTree._convert_node_to_json(self.right, "False") + "\n"
        json += "\t]\n"
        json += "}\n"
        return json

    @staticmethod
    def _convert_node_to_json(node, rule, count=2):
        json = "\t"*count + "{\n"
        if node.value is None:
            if len(node.class_probabilities) > 0:
                json += "\t"*count + "\"name\": \"" + str(node.label) + "( " + str(node.class_probabilities) + ")\",\n"
            else:
                json += "\t"*count + "\"name\": \"" + str(node.label) + " \",\n"
            json += "\t"*count + "\"rule\": \"" + rule + "\"\n"
        else:
            json += "\t"*count + "\"name\": \"" + str(node.label) + " <= " + str(node.value) + "\",\n"
            json += "\t"*count + "\"rule\": \"" + rule + "\",\n"
            json += "\t"*count + "\"children\": [\n"
            json += DecisionTree._convert_node_to_json(node.left, "True", count=count + 1) + ",\n"
            json += DecisionTree._convert_node_to_json(node.right, "False", count=count + 1) + "\n"
            json += "\t"*count + "]\n"
        json += "\t"*count + "}"

        return json

    @staticmethod
    def convert_from_json(json_file):
        """Converts a json file to a decision tree object

        **Params**
        ----------
          - `json_file` (string) - the JSON file to convert

        **Returns**
        -----------
            a decision tree object parsed from the JSON file
        """
        tree_json = json.loads(json_file)
        tree = DecisionTree()
        split_name = tree_json['name'].split(" <= ")
        label, value = split_name[0], split_name[1]
        tree.label = label
        tree.value = value
        tree.left = DecisionTree._convert_json_to_node(tree_json['children'][0])
        tree.right = DecisionTree._convert_json_to_node(tree_json['children'][1])
        return tree

    @staticmethod
    def _convert_json_to_node(_dict):
        tree = DecisionTree()
        split_name = _dict['name'].split(" <= ")
        if len(split_name) > 1:
            label, value = split_name[0], split_name[1]
            tree.label = label
            tree.value = value
            if 'children' in _dict:
                tree.left = DecisionTree._convert_json_to_node(_dict['children'][0])
                tree.right = DecisionTree._convert_json_to_node(_dict['children'][1])
        else:
            tree.label = split_name[0]
            tree.value = None
            tree.left = None
            tree.right = None
        return tree

    # </editor-fold>

    # <editor-fold desc="Probability estimation methods">

    @staticmethod
    def _init_tree(tree, labels):
        for label in np.unique(labels):
            tree.class_probabilities[str(int(label))] = 0.0

        if tree.value is not None:
            DecisionTree._init_tree(tree.left, labels)
            DecisionTree._init_tree(tree.right, labels)

    def populate_samples(self, feature_vectors, labels):
        """Use the given data to calculate probability estimates at each leaf

        **Params**
        ----------
          - `feature_vectors` (pandas DataFrame or list of dict) - the training feature vectors for each sample

          - `labels` (pandas Series or list) - the training class labels for each sample

        **Returns**
        -----------
            nothing, the decision tree object where this method was called on will be changed
        """
        index = 0
        DecisionTree._init_tree(self, np.unique(labels))
        for _index, feature_vector in feature_vectors.iterrows():
            current_node = self
            while current_node.value is not None:
                current_node.class_probabilities[str(int(labels[index]))] += 1
                if feature_vector[current_node.label] <= current_node.value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            current_node.class_probabilities[str(int(labels[index]))] += 1
            index += 1
        return

    # </editor-fold>

    # <editor-fold desc="Pruning methods">

    def _set_parents(self):
        if self.value is not None:
            self.left.parent = self
            self.left._set_parents()
            self.right.parent = self
            self.right._set_parents()

    def count_nodes(self):
        """Count the total number of nodes in the tree, used as metric for model complexity"""
        if self.value is None:
            return 1
        else:
            left = 0
            if self.left is not None: left = self.left.count_nodes()
            right = 0
            if self.right is not None: right = self.right.count_nodes()

            return left + right + 1

    def count_leaves(self):
        """Count the total number of leaves (`decisiontree.DecisionTree.value` = `None`) in the tree,
        used as metric for model complexity"""
        if self.value is None:
            return 1
        else:
            return self.left.count_leaves() + self.right.count_leaves()

    def _get_leaves(self):
        if self.value is None:
            return [self]
        else:
            leaves=[]
            leaves.extend(self.left._get_leaves())
            leaves.extend(self.right._get_leaves())
            return leaves

    def _get_nodes(self):
        if self.value is not None:
            nodes = [self]
        else:
            nodes = []
        if self.left is not None:
            nodes.extend(self.left._get_nodes())
        if self.right is not None:
            nodes.extend(self.right._get_nodes())
        return nodes

    def _calc_leaf_error(self, total_train_samples):
        return sum([(sum(leaf.class_probabilities.values()) / total_train_samples) *
                    (1 - leaf.class_probabilities[str(leaf.label)]/sum(leaf.class_probabilities.values()))
                    if sum(leaf.class_probabilities.values()) > 0 else 0
                    for leaf in self._get_leaves()])

    def _calc_node_error(self, total_train_samples):
        return (1 - max(self.class_probabilities.iteritems(), key=operator.itemgetter(1))[1]/sum(self.class_probabilities.values())) \
               * (sum(self.class_probabilities.values()) / total_train_samples)

    def _calculate_alpha(self, total_train_samples):
        if self.count_leaves() > 1:
            return (self._calc_node_error(total_train_samples) - self._calc_leaf_error(total_train_samples)) / (self.count_leaves() - 1)
        else:
            return (self._calc_node_error(total_train_samples) - self._calc_leaf_error(total_train_samples))

    def _calculate_cost_complexity(self, total_train_samples, alpha):
        return self._calc_leaf_error(total_train_samples) + alpha * self.count_leaves()

    def _prune_node(self, node, parents=[], directions=[], nodes=[], direction='left'):
        if self == node:
            # print'match'
            self_copy = copy(self)
            self.label = max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
            self.value = None
            self.right = None
            self.left = None
            # return self_copy, parent, direction
            parents.append(self.parent)
            directions.append(direction)
            nodes.append(self_copy)
        else:
            if self.left is not None and self.left.value is not None:
                self.left._prune_node(node, parents, directions, nodes, 'left')
            if self.right is not None and self.right.value is not None:
                self.right._prune_node(node, parents, directions, nodes, 'right')

    def _generate_subtree(self, total_train_samples, alphas={}):
        # print self.label, self.value
        if self.value is not None:
            calc_alpha = self._calculate_alpha(total_train_samples)
            alphas[self] = (calc_alpha, self.count_nodes())
            if self.left.value is not None:
                self.left._generate_subtree(total_train_samples, alphas)
            if self.right.value is not None:
                self.right._generate_subtree(total_train_samples, alphas)
        return alphas

    def _generate_subtree_sequence(self, total_train_samples):
        subtrees = {}
        current_tree = deepcopy(self)
        while current_tree.left is not None or current_tree.right is not None:
            generated_trees = current_tree._generate_subtree(total_train_samples, {})
            # print generated_trees.values()
            best = min(generated_trees.items(), key=operator.itemgetter(1))
            tree, alpha = best[0], best[1][0]
            current_tree._prune_node(tree)
            subtrees[deepcopy(current_tree)] = alpha
        return subtrees

    def _minimize_cost_complexity(self, total_train_samples, alpha):
        while 1:
            min_complexity, min_nodes = self._calculate_cost_complexity(total_train_samples, alpha), self.count_nodes()
            # print 'Can we improve?', (min_complexity, min_nodes)
            best_node_to_prune = None
            for node in self._get_nodes():
                # Make a copy of the node
                label_copy = node.label
                value_copy = node.value
                right_copy = node.right
                left_copy = node.left

                # Prune the node
                node.label = max(node.class_probabilities.items(), key=operator.itemgetter(1))[0]
                node.value = None
                node.right = None
                node.left = None

                # Calculate hypothetical cost complexity
                complexity, nodes = self._calculate_cost_complexity(total_train_samples, alpha), self.count_nodes()

                # Restore the node
                node.label = label_copy
                node.value = value_copy
                node.left = left_copy
                node.right = right_copy

                # We found a new best node?
                if (complexity, nodes) <= (min_complexity, min_nodes):
                    best_node_to_prune = node
                    min_complexity = complexity
                    min_nodes = nodes

            # Did we find a better node?
            if best_node_to_prune is not None:
                # print 'best_node:', best_node_to_prune.label, best_node_to_prune.value, min_complexity
                best_node_to_prune.label = max(best_node_to_prune.class_probabilities.items(), key=operator.itemgetter(1))[0]
                best_node_to_prune.value = None
                best_node_to_prune.right = None
                best_node_to_prune.left = None
            else:
                # print 'No new best node found'
                return self

    # </editor-fold>

    def _code_folding_does_not_work_in_pycharm_without_a_final_stub_method_here(self):
        pass