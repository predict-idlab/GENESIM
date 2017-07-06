from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

from constructors.ensemble import RFClassification
from data.load_all_datasets import load_all_datasets

import numpy as np

from decisiontree import DecisionTree

from refined_rf import RefinedRandomForest

rf = RFClassification()

NR_FOLDS = 5


def _convert_to_tree(dt, features):
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


for dataset in load_all_datasets():
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']

    skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)

    for fold, (train_idx, test_idx) in enumerate(skf):
        print 'Fold', fold+1, '/', NR_FOLDS, 'for dataset', dataset['name']
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        rf.construct_classifier(train, feature_cols, label_col)

        for estimator in rf.clf.estimators_:
            print estimator.tree_
            print _convert_to_tree(estimator, feature_cols)

        predictions = rf.evaluate_multiple(X_test).astype(int)
        conf_matrix = confusion_matrix(y_test, predictions)
        print conf_matrix
        diagonal_sum = sum(
            [conf_matrix[i][i] for i in range(len(conf_matrix))])
        norm_diagonal_sum = sum(
            [float(conf_matrix[i][i]) / float(sum(conf_matrix[i])) for i in
             range(len(conf_matrix))])
        total_count = np.sum(conf_matrix)
        print 'Accuracy:', float(diagonal_sum) / float(total_count)
        print 'Balanced accuracy:', float(norm_diagonal_sum) / float(conf_matrix.shape[0])