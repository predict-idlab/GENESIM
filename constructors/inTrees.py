"""
    inTrees / STEL
    --------------

    Merges different decision trees in an ensemble together in an ordered rule list

    Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.

    Reference:
        Houtao Deng
        "Interpreting Tree Ensembles with inTrees"
"""

import sys
import re

import numpy as np
import pandas as pd
import pandas.rpy.common as com
import rpy2.robjects as ro

from rpy2.robjects.packages import importr

from constructors import ensemble

sys.path.append('../')


class Condition:
    """
    Class which represents one part of the rule (which can be seen as a conjunction of conditions)
    """
    def __init__(self, feature, test, value):
        self.feature = feature
        '''The feature on which the test is performed'''
        self.test = test
        '''What kind of test is done. Must be either `==`, `>` or `<=`'''
        self.value = value
        '''The threshold value'''

    def evaluate(self, feature_vector):
        """Create a prediction for a sample (using its feature vector)

        **Params**
        ----------
          - `feature_vector` (pandas Series or dict) - the sample to evaluate, must be a `pandas Series` object or a
          `dict`. It is important that the attribute keys in the sample are the same as the labels occuring in the rules.

        **Returns**
        -----------
            `True` if feature_vector[<feature>] <test> <value>, where <test> is equal to `==`, `>` or `<=`
        """
        if self.value is None:
            return True
        elif self.test == '==':
            return feature_vector[self.feature] == self.value
        elif self.test == '>':
            return feature_vector[self.feature] > self.value
        else:
            return feature_vector[self.feature] <= self.value


class Rule:
    """
    Class which represents a rule, which is a conjunction of conditions
    """
    def __init__(self, index, conditions, prediction):
        self.index = index
        '''The index of this rule in a rule list (which is traversed sequentially until a match is found).'''
        self.rules = conditions
        '''A list of `constructors.inTrees.Condition`'''
        self.prediction = prediction
        '''This is returned when a sample fully complies to the rule (`True` for all conditions)'''

    def evaluate(self, feature_vector):
        """Create a prediction for a sample (using its feature vector)

        **Params**
        ----------
          - `feature_vector` (pandas Series or dict) - the sample to evaluate, must be a `pandas Series` object or a
          `dict`. It is important that the attribute keys in the sample are the same as the labels occuring in the rules.

        **Returns**
        -----------
            `True` if `True` for each condition in conditions
        """
        for rule in self.rules:
            if not rule.evaluate(feature_vector): return False, -1
        return True, self.prediction


class OrderedRuleList:
    """
    Class which represents a list of rules. To make a prediction, the list is traversed and when a rule is found where
    the sample complies to, its prediction is returned.
    """
    def __init__(self, rule_list):
        self.rule_list = rule_list
        '''A list of `constructors.inTrees.Rule`'''

    def _evaluate(self, feature_vector):
        for ruleset in sorted(self.rule_list, key=lambda x: x.index):  # Sort to make sure they are evaluated in order
            rule_evaluation_result, rule_evaluation_pred = ruleset.evaluate(feature_vector)
            if rule_evaluation_result: return rule_evaluation_pred
        return None

    def print_rules(self):
        """Print the rules"""
        for rule_set in self.rule_list:
            print '*' + ' & '.join([str(rule.feature)+' '+str(rule.test)+' '+str(rule.value) for rule in rule_set.rules]), '==>', rule_set.prediction

    def evaluate_multiple(self, feature_vectors):
        """Wrapper method to evaluate multiple vectors at once (just a for loop where evaluate is called)

        **Params**
        ----------
          - `feature_vectors` (pandas DataFrame or list of dicts) - the samples to evaluate

        **Returns**
        -----------
            a class label
        """
        results = []

        for _index, feature_vector in feature_vectors.iterrows():
            results.append(self._evaluate(feature_vector))

        return np.asarray(results)


class inTreesClassifier:

    def __init__(self):
        pass

    def _convert_to_r_dataframe(self, df, strings_as_factors=False):
        """
        Convert a pandas DataFrame to a R data.frame.

        Parameters
        ----------
        df: The DataFrame being converted
        strings_as_factors: Whether to turn strings into R factors (default: False)

        Returns
        -------
        A R data.frame

        """

        import rpy2.rlike.container as rlc

        columns = rlc.OrdDict()

        # FIXME: This doesn't handle MultiIndex

        for column in df:
            value = df[column]
            value_type = value.dtype.type

            if value_type == np.datetime64:
                value = com.convert_to_r_posixct(value)
            else:
                value = [item if pd.notnull(item) else com.NA_TYPES[value_type]
                         for item in value]

                value = com.VECTOR_TYPES[value_type](value)

                if not strings_as_factors:
                    I = ro.baseenv.get("I")
                    value = I(value)

            columns[column] = value

        r_dataframe = ro.DataFrame(columns)
        del columns

        r_dataframe.rownames = ro.StrVector(list(df.index))
        r_dataframe.colnames = list(df.columns)

        return r_dataframe

    def _tree_to_R_object(self, tree, feature_mapping):
        node_mapping = {}
        nodes = tree._get_nodes()
        nodes.extend(tree._get_leaves())
        for i, node in enumerate(nodes):
            node_mapping[node] = i+1
        vectors = []
        for node in nodes:
            if node.value is not None:
                vectors.append([node_mapping[node], node_mapping[node.left], node_mapping[node.right],
                                feature_mapping[node.label], node.value, 1, 0])
            else:
                vectors.append([node_mapping[node], 0, 0, 0, 0.0, -1, node.label])

        df = pd.DataFrame(vectors)
        df.columns = ['id', 'left daughter', 'right daughter', 'split var', 'split point', 'status', 'prediction']
        df = df.set_index('id')
        df.index.name = None

        return self._convert_to_r_dataframe(df)

    def construct_rule_list(self, train_df, label_col, tree_constructors, nr_bootstraps=3):
        """ Construct an `constructors.inTrees.OrderedRuleList` from an ensemble of decision trees

        **Params**
        ----------
          - `train_df` (pandas DataFrame) - the training data

          - `label_col` (string) - the column identifier for the class labels

          - `tree_constructors` (`constructors.treeconstructor.TreeConstructor`) - the decision tree induction algorithms used to create an ensemble with

          - `nr_bootstraps` (pandas DataFrame) - how many times do we apply bootstrapping for each TreeConstructor? The size of the ensemble will be equal to
          |tree_constructors|*nr_bootstraps

        **Returns**
        -----------
            an OrderedRuleList
        """
        y_train = train_df[label_col]
        X_train = train_df.copy()
        X_train = X_train.drop(label_col, axis=1)

        importr('randomForest')
        importr('inTrees')

        ro.globalenv["X"] = com.convert_to_r_dataframe(X_train)
        ro.globalenv["target"] = ro.FactorVector(y_train.values.tolist())

        feature_mapping = {}
        feature_mapping_reverse = {}
        for i, feature in enumerate(X_train.columns):
            feature_mapping[feature] = i + 1
            feature_mapping_reverse[i + 1] = feature

        treeList = []
        for tree in ensemble.bootstrap(train_df, label_col, tree_constructors, nr_classifiers=nr_bootstraps):
            if tree.count_nodes() > 1: treeList.append(self._tree_to_R_object(tree, feature_mapping))

        ro.globalenv["treeList"] = ro.Vector([len(treeList), ro.Vector(treeList)])
        ro.r('names(treeList) <- c("ntree", "list")')

        rules = ro.r('buildLearner(getRuleMetric(extractRules(treeList, X), X, target), X, target)')
        rules=list(rules)
        conditions=rules[int(0.6*len(rules)):int(0.8*len(rules))]
        predictions=rules[int(0.8*len(rules)):]

        # Create a OrderedRuleList
        rulesets = []
        for idx, (condition, prediction) in enumerate(zip(conditions, predictions)):
            # Split each condition in Rules to form a RuleSet
            rulelist = []
            condition_split = [x.lstrip().rstrip() for x in condition.split('&')]
            for rule in condition_split:
                feature = feature_mapping_reverse[int(re.findall(r',[0-9]+]', rule)[0][1:-1])]

                lte = re.findall(r'<=', rule)
                gt = re.findall(r'>', rule)
                eq = re.findall(r'==', rule)
                cond = lte[0] if len(lte) else (gt[0] if len(gt) else eq[0])

                extract_value = re.findall(r'[=>]-?[0-9\.]+', rule)
                if len(extract_value):
                    value = float(re.findall(r'[=>]-?[0-9\.]+', rule)[0][1:])
                else:
                    feature = 'True'
                    value = None

                rulelist.append(Condition(feature, cond, value))
            rulesets.append(Rule(idx, rulelist, prediction))

        return OrderedRuleList(rulesets)