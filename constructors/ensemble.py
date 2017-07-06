"""
Contains wrappers around well-known ensemble techniques: Random Forest and XGBoost.

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

import time
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import decisiontree


class EnsembleConstructor(object):
    """This class is an interface for all tree induction algorithms."""

    def __init__(self):
        """In the init method, all hyper-parameters should be set."""
        self.clf = None

    def get_name(self):
        """Get the name of the induction algorithm implemented."""
        raise NotImplementedError("This method needs to be implemented")

    def construct_classifier(self, train, features, label_col):
        """Construct an ensemble classifier.

        **Params**
        ----------
          - `train` (pandas DataFrame) - a `Dataframe` containing all the training data

          - `features` (pandas Series or list) - the names of the feature columns

          - `label_col` (string) - the name of the class label column

        **Returns**
        -----------
            an ensemble classifier
        """
        raise NotImplementedError("This method needs to be implemented")

    def evaluate_multiple(self, feature_vectors):
        """Evaluate multiple samples

        **Params**
        ----------
          - `feature_vectors` (pandas DataFrame) - a `Dataframe` containing all the feature vectors

        **Returns**
        -----------
            a list of predicted class labels

        """
        return self.clf.predict(feature_vectors)


class XGBClassification(EnsembleConstructor):

    def get_name(self):
        return 'XGBoost'

    def __init__(self):
        super(XGBClassification, self).__init__()
        self.nr_clf = 0
        self.time = 0

    def construct_classifier(self, train, features, label_col):
        data = train[features]
        target = train[label_col]

        def xgbcv(nr_classifiers, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma,
                  reg_lambda):
            nr_classifiers = int(nr_classifiers)
            max_depth = int(max_depth)
            min_child_weight = int(min_child_weight)
            return cross_val_score(XGBClassifier(learning_rate=learning_rate, n_estimators=nr_classifiers,
                                                 gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
                                                 nthread=1, scale_pos_weight=1, reg_lambda=reg_lambda,
                                                 min_child_weight=min_child_weight, max_depth=max_depth),
                                   data, target, 'accuracy', cv=5).mean()

        params = {
            'nr_classifiers': (50, 1000),
            'learning_rate': (0.01, 0.3),
            'max_depth': (5, 10),
            'min_child_weight': (2, 10),
            'subsample': (0.7, 0.8),
            'colsample_bytree': (0.5, 0.99),
            'gamma': (1., 0.01),
            'reg_lambda': (0, 1)
        }

        xgbBO = BayesianOptimization(xgbcv, params, verbose=0)
        xgbBO.maximize(init_points=10, n_iter=20, n_restarts_optimizer=100)
        # xgbBO.maximize(init_points=1, n_iter=1, n_restarts_optimizer=100)

        best_params = xgbBO.res['max']['max_params']

        best_nr_classifiers = int(best_params['nr_classifiers'])
        self.nr_clf = best_nr_classifiers
        best_max_depth = int(best_params['max_depth'])
        best_min_child_weight = int(best_params['min_child_weight'])
        best_colsample_bytree = best_params['colsample_bytree']
        best_subsample = best_params['subsample']
        best_reg_lambda = best_params['reg_lambda']
        best_learning_rate = best_params['learning_rate']
        best_gamma = best_params['gamma']

        print(best_nr_classifiers)

        self.clf = XGBClassifier(learning_rate=best_learning_rate, n_estimators=best_nr_classifiers,
                                 gamma=best_gamma, subsample=best_subsample, colsample_bytree=best_colsample_bytree,
                                 nthread=1, scale_pos_weight=1, reg_lambda=best_reg_lambda,
                                 min_child_weight=best_min_child_weight, max_depth=best_max_depth)
        start = time.time()
        self.clf.fit(data, target)
        self.time = time.time() - start

        return self

    def evaluate_multiple(self, feature_vectors):
        return self.clf.predict(feature_vectors)


class RFClassification(EnsembleConstructor):

    def get_name(self):
        return 'RF'

    def __init__(self):
        super(RFClassification, self).__init__()
        self.nr_clf = 0
        self.time = 0

    def construct_classifier(self, train, features, label_col):
        data = train[features]
        target = train[label_col]

        def rfcv(nr_classifiers, max_depth, min_samples_leaf, bootstrap, criterion, max_features):
            nr_classifiers = int(nr_classifiers)
            max_depth = int(max_depth)
            min_samples_leaf = int(min_samples_leaf)
            if np.round(bootstrap):
                bootstrap = True
            else:
                bootstrap = False
            if np.round(criterion):
                criterion = 'gini'
            else:
                criterion = 'entropy'
            if np.round(max_features):
                max_features = None
            else:
                max_features = 1.0

            return cross_val_score(RandomForestClassifier(n_estimators=nr_classifiers, max_depth=max_depth,
                                                          min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                                          criterion=criterion, max_features=max_features),
                                   data, target, 'accuracy', cv=5).mean()

        params = {
            'nr_classifiers': (10, 1000),
            'max_depth': (5, 10),
            'min_samples_leaf': (2, 10),
            'bootstrap': (0, 1),
            'criterion': (0, 1),
            'max_features': (0, 1)
        }

        rfBO = BayesianOptimization(rfcv, params, verbose=0)
        rfBO.maximize(init_points=10, n_iter=20, n_restarts_optimizer=50)
        # rfBO.maximize(init_points=1, n_iter=1, n_restarts_optimizer=50)

        best_params = rfBO.res['max']['max_params']

        best_nr_classifiers = int(best_params['nr_classifiers'])
        self.nr_clf = best_nr_classifiers
        best_max_depth = int(best_params['max_depth'])
        best_min_samples_leaf = int(best_params['min_samples_leaf'])
        best_bootstrap = best_params['bootstrap']
        best_criterion = best_params['criterion']
        best_max_features = best_params['max_features']

        if np.round(best_bootstrap):
            best_bootstrap = True
        else:
            best_bootstrap = False
        if np.round(best_criterion):
            best_criterion = 'gini'
        else:
            best_criterion = 'entropy'
        if np.round(best_max_features):
            best_max_features = None
        else:
            best_max_features = 1.0

        self.clf = RandomForestClassifier(n_estimators=best_nr_classifiers, max_depth=best_max_depth,
                                          min_samples_leaf=best_min_samples_leaf, bootstrap=best_bootstrap,
                                          criterion=best_criterion, max_features=best_max_features)
        start = time.time()
        self.clf.fit(data, target)

        self.time = time.time() - start

        return self

    def evaluate_multiple(self, feature_vectors):
        return self.clf.predict(feature_vectors)


def bootstrap(data, class_label, tree_constructors, bootstrap_features=False, nr_classifiers=3, boosting=True):
    """
    Bootstrapping ensemble technique

    **Params**
    ----------
    - `data` (DataFrame): containing all the data to be bootstrapped

    - `class_label` (string): the column in the dataframe that contains the target variables

    - `tree_constructors` (list): the induction algorithms (`constructors.treeconstructor.TreeConstructor`) used

    - `bootstrap_features` (boolean): if `True`, then apply bootstrapping to the features as well

    - `nr_classifiers` (int): for each `tree_constructor`, how many times must we bootstrap

    - `boosting` (boolean): if `True`, then do create models with AdaBoost too

    **Returns**
    -----------
        a  vector of fitted classifiers, converted to DecisionTree (`decisiontree.DecisionTree`)
    """

    def _convert_to_tree(classifier, features):
        n_nodes = classifier.tree_.node_count
        children_left = classifier.tree_.children_left
        children_right = classifier.tree_.children_right
        feature = classifier.tree_.feature
        threshold = classifier.tree_.threshold
        classes = classifier.classes_

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
                decision_trees[i].label = classes[np.argmax(classifier.tree_.value[i][0])]
                decision_trees[i].value = None
            else:
                decision_trees[i].label = features[feature[i]]
                decision_trees[i].value = threshold[i]
        return decision_trees[0]

    idx = np.random.randint(0, len(data), (nr_classifiers, len(data)))
    decision_trees = []

    if boosting:
        ada = AdaBoostClassifier(base_estimator=None, n_estimators=nr_classifiers, learning_rate=0.25, random_state=1337)
        X_train = data.drop(class_label, axis=1).reset_index(drop=True)
        y_train = data[class_label].reset_index(drop=True)
        ada.fit(X_train, y_train)
        for estimator in ada.estimators_:
            dt = _convert_to_tree(estimator, X_train.columns)
            dt.data = data
            dt.populate_samples(X_train, y_train)
            decision_trees.append(dt)

    for indices in idx:
        if bootstrap_features:
            features = list(set(np.random.randint(0, len(data.columns), (1, len(data.columns))).tolist()[0]))
            X_bootstrap = data.iloc[indices, features].reset_index(drop=True)
            if class_label in X_bootstrap.columns:
                X_bootstrap = X_bootstrap.drop(class_label, axis=1)
            y_bootstrap = data.iloc[indices][class_label].reset_index(drop=True)
        else:
            X_bootstrap = data.iloc[indices, :].drop(class_label, axis=1).reset_index(drop=True)
            y_bootstrap = data.iloc[indices][class_label].reset_index(drop=True)

        X = data.drop(class_label, axis=1).reset_index(drop=True)
        y = data[class_label].reset_index(drop=True)
        train_bootstrap = X_bootstrap.copy()
        train_bootstrap[y_bootstrap.name] = y_bootstrap

        for tree_constructor in tree_constructors:
            tree = tree_constructor.construct_classifier(train_bootstrap, X_bootstrap.columns, y_bootstrap.name)
            # print 'Number of nodes in stub:', tree_constructor.get_name(), count_nodes(tree)
            # print tree_constructor.get_name(), tree.count_nodes()
            tree.data = data.iloc[indices, :].reset_index(drop=True)
            tree.populate_samples(X, y)
            decision_trees.append(tree)

    return decision_trees