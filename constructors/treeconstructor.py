"""
Contains wrappers around well-known decision tree induction algorithms: C4.5, CART, QUEST and GUIDE.

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import Orange
import operator
import os
import time
import subprocess

import decisiontree


class TreeConstructor(object):
    """This class is an interface for all tree induction algorithms."""

    def __init__(self):
        """In the init method, all hyper-parameters should be set."""
        pass

    def get_name(self):
        """Get the name of the induction algorithm implemented."""
        raise NotImplementedError("This method needs to be implemented")

    def construct_classifier(self, train, features, label_col):
        """Construct a `decisiontree.DecisionTree` object from the given training data

        **Params**
        ----------
          - `train` (pandas DataFrame) - a `Dataframe` containing all the training data

          - `features` (list) - the names of the feature columns

          - `label_col` (string) - the name of the class label column

        **Returns**
        -----------
            a DecisionTree object
        """
        raise NotImplementedError("This method needs to be implemented")


# <editor-fold desc="Conversion methods between Orange and pandas">
def _series2descriptor(d, discrete=False):
    if d.dtype is np.dtype("float"):
        return Orange.feature.Continuous(str(d.name))
    elif d.dtype is np.dtype("int"):
        return Orange.feature.Continuous(str(d.name), number_of_decimals=0)
    else:
        t = d.unique()
        if discrete or len(t) < len(d) / 2:
            t.sort()
            return Orange.feature.Discrete(str(d.name), values=list(t.astype("str")))
        else:
            return Orange.feature.String(str(d.name))


def _df2domain(df):
    featurelist = [_series2descriptor(df.iloc[:, col]) for col in xrange(len(df.columns))]
    return Orange.data.Domain(featurelist)


def _df2table(df):
    # It seems they are using native python object/lists internally for Orange.data types (?)
    # And I didn't find a constructor suitable for pandas.DataFrame since it may carry
    # multiple dtypes
    #  --> the best approximate is Orange.data.Table.__init__(domain, numpy.ndarray),
    #  --> but the dtype of numpy array can only be "int" and "float"
    #  -->  * refer to src/orange/lib_kernel.cpp 3059:
    #  -->  *    if (((*vi)->varType != TValue::INTVAR) && ((*vi)->varType != TValue::FLOATVAR))
    #  --> Documents never mentioned >_<
    # So we use numpy constructor for those int/float columns, python list constructor for other

    tdomain = _df2domain(df)
    ttables = [_series2table(df.iloc[:, i], tdomain[i]) for i in xrange(len(df.columns))]
    return Orange.data.Table(ttables)


def _series2table(series, variable):
    if series.dtype is np.dtype("int") or series.dtype is np.dtype("float"):
        # Use numpy
        # Table._init__(Domain, numpy.ndarray)
        return Orange.data.Table(Orange.data.Domain(variable), series.values[:, np.newaxis])
    else:
        # Build instance list
        # Table.__init__(Domain, list_of_instances)
        tdomain = Orange.data.Domain(variable)
        tinsts = [Orange.data.Instance(tdomain, [i]) for i in series]
        return Orange.data.Table(tdomain, tinsts)
        # 5x performance


def _column2df(col):
    if type(col.domain[0]) is Orange.feature.Continuous:
        return (col.domain[0].name, pd.Series(col.to_numpy()[0].flatten()))
    else:
        tmp = pd.Series(np.array(list(col)).flatten())  # type(tmp) -> np.array( dtype=list (Orange.data.Value) )
        tmp = tmp.apply(lambda x: str(x[0]))
        return (col.domain[0].name, tmp)


def _table2df(tab):
    # Orange.data.Table().to_numpy() cannot handle strings
    # So we must build the array column by column,
    # When it comes to strings, python list is used
    series = [_column2df(tab.select(i)) for i in xrange(len(tab.domain))]
    series_name = [i[0] for i in series]  # To keep the order of variables unchanged
    series_data = dict(series)
    return pd.DataFrame(series_data, columns=series_name)

# </editor-fold>


class C45Constructor(TreeConstructor):
    """This class contains an implementation of C4.5, written by Quinlan. It uses an extern library
    for this called [Orange](http://docs.orange.biolab.si/2/reference/rst/Orange.classification.tree.html#Orange.classification.tree.C45Learner)."""

    def __init__(self, gain_ratio=False, cf=0.15):
        super(C45Constructor, self).__init__()
        self.gain_ratio = gain_ratio
        '''boolean value that indicates if either gain ratio or information gain is used as split metric'''
        self.cf = cf
        '''pruning confidence level: the lower this value, the more pruning will be done'''

    def get_name(self):
        return "C4.5"

    def construct_classifier(self, train, features, label_col, param_opt=True):
        training_feature_vectors = train[features].copy()
        labels = train[label_col].copy()
        if param_opt:
            optimal_clf = C45Constructor.get_best_c45_classifier(train, label_col,
                                                                 StratifiedKFold(train[label_col], n_folds=3,
                                                                                 shuffle=True, random_state=None))
            self.cf = optimal_clf.cf

        # First call df2table on the feature table
        orange_feature_table = _df2table(training_feature_vectors)

        # Convert classes to strings and call df2table
        orange_labels_table = _df2table(pd.DataFrame(labels.map(str)))

        # Merge two tables
        orange_table = Orange.data.Table([orange_feature_table, orange_labels_table])

        return self._orange_dt_to_my_dt(Orange.classification.tree.C45Learner(orange_table, gain_ratio=self.gain_ratio,
                                                                              cf=self.cf, min_objs=2, subset=False).tree)

    def _orange_dt_to_my_dt(self, orange_dt_root):
        # Check if leaf
        if orange_dt_root.node_type == Orange.classification.tree.C45Node.Leaf:
            return decisiontree.DecisionTree(left=None, right=None, label=str(int(orange_dt_root.leaf)), data=None, value=None)
        else:
            dt = decisiontree.DecisionTree(label=orange_dt_root.tested.name, data=None, value=orange_dt_root.cut)
            dt.left = self._orange_dt_to_my_dt(orange_dt_root.branch[0])
            dt.right = self._orange_dt_to_my_dt(orange_dt_root.branch[1])
            return dt

    @staticmethod
    def get_best_c45_classifier(train, label_col, skf_tune):
        """Returns a `treeconstructor.C45Constructor` with optimized hyper-parameters using
        [Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)

        **Params**
        ----------
          - `train` (pandas DataFrame) - `a pandas Dataframe` with all training data

          - `label_col` (string) - the column identifier for the label in the `train` Dataframe

          - `skf_tune` (`sklearn.cross_validation.StratifiedKFold`) - cross-validation object to tune parameters

        **Returns**
        -----------
            a C45Constructor with optimized hyper-parameters
        """
        c45 = C45Constructor()
        cfs = np.arange(0.05, 1.05, 0.05)
        cfs_errors = {}
        for cf in cfs:  cfs_errors[cf] = []

        for train_tune_idx, val_tune_idx in skf_tune:
            train_tune = train.iloc[train_tune_idx, :]
            X_train_tune = train_tune.drop(label_col, axis=1)
            y_train_tune = train_tune[label_col]
            val_tune = train.iloc[val_tune_idx, :]
            X_val_tune = val_tune.drop(label_col, axis=1)
            y_val_tune = val_tune[label_col]
            for cf in cfs:
                c45.cf = cf
                tree = c45.construct_classifier(train_tune, X_train_tune.columns, label_col, param_opt=False)
                predictions = tree.evaluate_multiple(X_val_tune).astype(int)
                cfs_errors[cf].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))

        for cf in cfs:
            cfs_errors[cf] = np.mean(cfs_errors[cf])

        c45.cf = min(cfs_errors.items(), key=operator.itemgetter(1))[0]
        return c45


class CARTConstructor(TreeConstructor):
    """This class contains an implementation of CART, written by Breiman. It uses an extern library
    for this called [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)."""

    def __init__(self, criterion='gini', min_samples_leaf=1, min_samples_split=2, max_depth=10):
        super(CARTConstructor, self).__init__()
        self.min_samples_leaf = min_samples_leaf
        '''pre-prune condition: when the current number of samples is lower than this threshold, then stop'''
        self.min_samples_split = min_samples_split
        '''pre-prune condition: when a split causes the number of samples in one of the two partitions to be lower
        than this threshold, then stop'''
        self.max_depth = max_depth
        '''pre-prune condition: when a depth equal to this parameter is reached, then stop'''
        self.criterion = criterion
        '''defines which split criterion to use, is either equal to `gini` or `entropy`'''

    def get_name(self):
        return "CART"

    def construct_classifier(self, train, features, label_col, param_opt=True):
        training_feature_vectors = train[features]
        labels = train[label_col]
        train = training_feature_vectors.copy()
        label_col = labels.name
        train[label_col] = labels
        if param_opt:
            optimal_clf = CARTConstructor.get_best_cart_classifier(train, label_col,
                                                                 StratifiedKFold(train[label_col], n_folds=3,
                                                                                 shuffle=True, random_state=None))
            self.max_depth = optimal_clf.max_depth
            self.min_samples_split = optimal_clf.min_samples_split

        self.features = list(training_feature_vectors.columns)

        self.y = labels.values
        self.X = training_feature_vectors[self.features]


        self.dt = DecisionTreeClassifier(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                         min_samples_split=self.min_samples_split, max_depth=self.max_depth)
        self.dt.fit(self.X, self.y)

        return self._convert_to_tree()

    def _convert_to_tree(self):
        """Convert a sklearn object to a `decisiontree.decisiontree` object"""
        n_nodes = self.dt.tree_.node_count
        children_left = self.dt.tree_.children_left
        children_right = self.dt.tree_.children_right
        feature = self.dt.tree_.feature
        threshold = self.dt.tree_.threshold
        classes = self.dt.classes_

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes)
        decision_trees = [None] * n_nodes
        for i in range(n_nodes):
            decision_trees[i] = decisiontree.DecisionTree()
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
                decision_trees[i].label = self.dt.classes_[np.argmax(self.dt.tree_.value[i][0])]
                decision_trees[i].value = None
            else:
                decision_trees[i].label = self.features[feature[i]]
                decision_trees[i].value = threshold[i]

        return decision_trees[0]

    @staticmethod
    def get_best_cart_classifier(train, label_col, skf_tune):
        """Returns a `treeconstructor.CARTConstructor` with optimized hyper-parameters using
        [Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)

        **Params**
        ----------
          - `train` (pandas DataFrame) - `a pandas Dataframe` with all training data

          - `label_col` (string) - the column identifier for the label in the `train` Dataframe

          - `skf_tune` (`sklearn.cross_validation.StratifiedKFold`) - cross-validation object to tune parameters

        **Returns**
        -----------
            a CARTConstructor with optimized hyper-parameters
        """
        cart = CARTConstructor()
        max_depths = np.arange(1,21,2)
        max_depths = np.append(max_depths, None)
        min_samples_splits = np.arange(2,20,1)

        errors = {}
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                errors[(max_depth, min_samples_split)] = []

        for train_tune_idx, val_tune_idx in skf_tune:
            train_tune = train.iloc[train_tune_idx, :]
            X_train_tune = train_tune.drop(label_col, axis=1)
            y_train_tune = train_tune[label_col]
            val_tune = train.iloc[val_tune_idx, :]
            X_val_tune = val_tune.drop(label_col, axis=1)
            y_val_tune = val_tune[label_col]
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    cart.max_depth = max_depth
                    cart.min_samples_split = min_samples_split
                    tree = cart.construct_classifier(train_tune, X_train_tune.columns, label_col, param_opt=False)
                    predictions = tree.evaluate_multiple(X_val_tune).astype(int)
                    errors[((max_depth, min_samples_split))].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))


        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                errors[(max_depth, min_samples_split)] = np.mean(errors[(max_depth, min_samples_split)])

        best_params = min(errors.items(), key=operator.itemgetter(1))[0]
        cart.max_depth = best_params[0]
        cart.min_samples_split = best_params[1]

        return cart


class QUESTConstructor(TreeConstructor):
    """This class contains a wrapper around an implementation of [QUEST](http://www.stat.wisc.edu/~loh/quest.html),
    written by Loh."""

    def __init__(self):
        super(QUESTConstructor, self).__init__()

    def get_name(self):
        return "QUEST"

    def construct_classifier(self, train, features, label_col):
        training_feature_vectors = train[features]
        labels = train[label_col]
        self._create_desc_and_data_file(training_feature_vectors, labels)
        input = open("in.txt", "w")
        output = file('out.txt', 'w')
        p = subprocess.Popen(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+'/quest > log.txt', stdin=subprocess.PIPE, shell=True)
        p.stdin.write("2\n")
        p.stdin.write("in.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("out.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("dsc.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("\n")
        p.wait()
        input.close()
        output.close()

        while not os.path.exists('in.txt'):
            time.sleep(1)
        p = subprocess.Popen(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+'/quest < in.txt > log.txt', stdin=subprocess.PIPE, shell=True)
        p.wait()

        output = file('out.txt', 'r')
        lines = output.readlines()
        output.close()

        start_index, end_index, counter = 0, 0, 0
        for line in lines:
            if line == '  Classification tree:\n':
                start_index = counter+2
            if line == '  Information for each node:\n':
                end_index = counter-1
            counter += 1
        tree = self._decision_tree_from_text(lines[start_index:end_index])

        self._remove_files()

        return tree

    def _decision_tree_from_text(self, lines):
        dt = decisiontree.DecisionTree()

        if '<=' in lines[0] or '>' in lines[0]:
            # Intermediate node
            node_name = lines[0].split(':')[0].lstrip()
            label, value = lines[0].split(':')[1].split('<=')
            label = ' '.join(label.lstrip().rstrip().split('.'))
            value = value.lstrip().split()[0]
            dt.label = label
            dt.value = float(value)
            dt.left = self._decision_tree_from_text(lines[1:])
            counter = 1
            while lines[counter].split(':')[0].lstrip() != node_name: counter+=1
            dt.right = self._decision_tree_from_text(lines[counter + 1:])
        else:
            # Terminal node
            dt.label = int(eval(lines[0].split(':')[1].lstrip()))

        return dt

    def _create_desc_and_data_file(self, training_feature_vectors, labels):
        dsc = open("dsc.txt", "w")
        data = open("data.txt", "w")

        dsc.write("data.txt\n")
        dsc.write("\"?\"\n")
        dsc.write("column, var, type\n")
        count = 1
        for col in training_feature_vectors.columns:
            dsc.write(str(count) + ' \"' + str(col) + '\" n\n')
            count += 1
        dsc.write(str(count) + ' ' + str(labels.name) + ' d')

        for i in range(len(training_feature_vectors)):
            sample = training_feature_vectors.iloc[i,:]
            for col in training_feature_vectors.columns:
                data.write(str(sample[col]) + ' ')
            if i != len(training_feature_vectors)-1:
                data.write(str(labels[i])+'\n')
            else:
                data.write(str(labels[i]))

        data.close()
        dsc.close()

    def _remove_files(self):
        os.remove('data.txt')
        os.remove('in.txt')
        os.remove('dsc.txt')
        os.remove('out.txt')
        os.remove('log.txt')


class GUIDEConstructor(TreeConstructor):
    """This class contains a wrapper around an implementation of [GUIDE](http://www.stat.wisc.edu/~loh/guide.html),
    written by Loh."""

    def __init__(self):
        super(GUIDEConstructor, self).__init__()

    def get_name(self):
        return "GUIDE"

    def construct_classifier(self, train, features, label_col):
        training_feature_vectors = train[features]
        labels = train[label_col]
        self._create_desc_and_data_file(training_feature_vectors, labels)
        input = open("in.txt", "w")
        output = file('out.txt', 'w')
        p = subprocess.Popen(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+'/guide > log.txt', stdin=subprocess.PIPE, shell=True)
        p.stdin.write("1\n")
        p.stdin.write("in.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("out.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("2\n")
        p.stdin.write("1\n")
        p.stdin.write("3\n")
        p.stdin.write("1\n")
        p.stdin.write('dsc.txt\n')
        p.stdin.write("\n")
        p.stdin.write("\n")
        p.stdin.write("\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("\n")
        p.stdin.write("\n")
        p.stdin.write("\n")
        p.stdin.write("2\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("\n")
        p.wait()
        input.close()
        output.close()

        while not os.path.exists('in.txt'):
            time.sleep(1)
        p = subprocess.Popen(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+'/guide < in.txt > log.txt', shell=True)
        p.wait()

        output = file('out.txt', 'r')
        lines = output.readlines()
        output.close()

        start_index, end_index, counter = 0, 0, 0
        for line in lines:
            if line == ' Classification tree:\n':
                start_index = counter+2
            if line == ' ***************************************************************\n':
                end_index = counter-1
            counter += 1
        tree = self._decision_tree_from_text(lines[start_index:end_index])

        # self.remove_files()

        # tree.visualise('GUIDE')
        return tree

    def _decision_tree_from_text(self, lines):

        dt = decisiontree.DecisionTree()

        if '<=' in lines[0] or '>' in lines[0] or '=' in lines[0]:
            # Intermediate node
            node_name = lines[0].split(':')[0].lstrip()
            # print(lines[0])
            label, value = lines[0].split(':')[1].split('<=')
            label = ' '.join(label.lstrip().rstrip().split('.'))
            value = value.lstrip().split()[0]
            dt.label = label
            dt.value = float(value)
            dt.left = self._decision_tree_from_text(lines[1:])
            counter = 1
            while lines[counter].split(':')[0].lstrip() != node_name: counter+=1
            dt.right = self._decision_tree_from_text(lines[counter + 1:])
        else:
            # Terminal node
            # print lines[0]
            dt.label = int(lines[0].split(':')[1].lstrip().split('.')[0])

        return dt

    def _create_desc_and_data_file(self, training_feature_vectors, labels):
        dsc = open("dsc.txt", "w")
        data = open("data.txt", "w")
        dsc.write("data.txt\n")
        dsc.write("\"?\"\n")
        dsc.write("1\n")
        count = 1
        for col in training_feature_vectors.columns:
            dsc.write(str(count) + ' \"' + str(col) + '\" n\n')
            count += 1
        dsc.write(str(count) + ' ' + str(labels.name) + ' d')

        for i in range(len(training_feature_vectors)):
            sample = training_feature_vectors.iloc[i,:]
            for col in training_feature_vectors.columns:
                data.write(str(sample[col]) + ' ')
            if i != len(training_feature_vectors)-1:
                data.write(str(labels[i])+'\n')
            else:
                data.write(str(labels[i]))

        data.close()
        dsc.close()

    def _remove_files(self):
        os.remove('data.txt')
        os.remove('in.txt')
        os.remove('dsc.txt')
        os.remove('out.txt')
        os.remove('log.txt')