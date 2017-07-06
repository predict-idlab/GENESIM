"""
This is an example script that will apply k-cross-validation on all datasets with a load function in
`data.load_datasets` and for all implemented tree constructors, ensemble techniques and GENESIM. In the end,
a confusion matrices will be stored at path `output/dataset_name_CVk.png` and the average model complexity and
computational time required for each of the algorithms will be printed out.

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""


import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold, KFold

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

import constructors.ISM
from constructors.ensemble import RFClassification, XGBClassification, bootstrap
from constructors.genesim import GENESIM
from constructors.inTrees import inTreesClassifier
from constructors.treeconstructor import QUESTConstructor, GUIDEConstructor, C45Constructor, CARTConstructor
from data.load_all_datasets import load_all_datasets
from decisiontree import DecisionTree

if __name__ == "__main__":

    algorithms = {QUESTConstructor().get_name(): QUESTConstructor(),
                  GUIDEConstructor().get_name(): GUIDEConstructor(),
                  CARTConstructor().get_name(): CARTConstructor(), C45Constructor().get_name(): C45Constructor(),
                  RFClassification().get_name(): RFClassification(),
                  XGBClassification().get_name(): XGBClassification()
                 }
    genesim = GENESIM()
    inTrees_clf = inTreesClassifier()

    NR_FOLDS = 5
    for dataset in load_all_datasets():
        df = dataset['dataframe']
        label_col = dataset['label_col']
        feature_cols = dataset['feature_cols']

        conf_matrices, avg_nodes, times = {}, {}, {}

        for algorithm in algorithms:
            conf_matrices[algorithm] = []
            avg_nodes[algorithm] = []
            times[algorithm] = []
        conf_matrices['GENESIM'], avg_nodes['GENESIM'], times['GENESIM'] = [], [], []
        conf_matrices['ISM'], avg_nodes['ISM'], times['ISM'] = [], [], []
        conf_matrices['inTrees'], avg_nodes['inTrees'], times['inTrees'] = [], [], []

        skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)

        for fold, (train_idx, test_idx) in enumerate(skf):
            print 'Fold', fold+1, '/', NR_FOLDS, 'for dataset', dataset['name']
            train = df.iloc[train_idx, :].reset_index(drop=True)
            X_train = train.drop(label_col, axis=1)
            y_train = train[label_col]
            test = df.iloc[test_idx, :].reset_index(drop=True)
            X_test = test.drop(label_col, axis=1)
            y_test = test[label_col]

            for algorithm in algorithms:
                print algorithm
                start = time.time()
                clf = algorithms[algorithm].construct_classifier(train, feature_cols, label_col)
                end = time.time()
                times[algorithm].append(end-start)
                predictions = clf.evaluate_multiple(X_test).astype(int)
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
                conf_matrices[algorithm].append(confusion_matrix(y_test, predictions))
                if type(clf) is DecisionTree:
                    avg_nodes[algorithm].append(clf.count_nodes())
                else:
                    avg_nodes[algorithm].append(clf.nr_clf)

            _constructors = [CARTConstructor(), QUESTConstructor(), GUIDEConstructor()]

            print 'inTrees'
            start = time.time()
            orl = inTrees_clf.construct_rule_list(train, label_col, _constructors, nr_bootstraps=25)
            end = time.time()
            times['inTrees'].append(end-start)
            predictions = orl.evaluate_multiple(X_test).astype(int)
            conf_matrices['inTrees'].append(confusion_matrix(y_test, predictions))
            conf_matrix = confusion_matrix(y_test, predictions)
            print conf_matrix
            diagonal_sum = sum(
                [conf_matrix[i][i] for i in range(len(conf_matrix))])
            norm_diagonal_sum = sum(
                [float(conf_matrix[i][i]) / float(sum(conf_matrix[i])) for i in
                 range(len(conf_matrix))])
            total_count = np.sum(conf_matrix)
            correct = 0
            for i in range(len(conf_matrix)):
                correct += conf_matrix[i][i] + conf_matrix[i][max(i - 1, 0)] * ((i - 1) >= 0) + \
                           conf_matrix[i][min(i + 1, len(conf_matrix[i]) - 1)] * ((i + 1) <= len(conf_matrix[i]) - 1)
            # print 'Accuracy [-1, +1]:', float(correct) / float(total_count)
            print 'Accuracy:', float(diagonal_sum) / float(total_count)
            print 'Balanced accuracy:', float(norm_diagonal_sum) / float(conf_matrix.shape[0])
            avg_nodes['inTrees'].append(len(orl.rule_list))

            print 'ISM'
            start = time.time()
            ism_tree = constructors.ISM.ism(bootstrap(train, label_col, _constructors, boosting=True, nr_classifiers=5),
                                            train, label_col, min_nr_samples=1, calc_fracs_from_ensemble=False)
            ism_pruned = ism_tree.cost_complexity_pruning(X_train, y_train, 'ism', ism_constructors=_constructors,
                                                          ism_calc_fracs=False, n_folds=3, ism_nr_classifiers=5,
                                                          ism_boosting=True)
            end = time.time()
            times['ISM'].append(end - start)
            predictions = ism_pruned.evaluate_multiple(X_test).astype(int)
            conf_matrices['ISM'].append(confusion_matrix(y_test, predictions))
            avg_nodes['ISM'].append(ism_pruned.count_nodes())
            conf_matrix = confusion_matrix(y_test, predictions)
            print conf_matrix
            diagonal_sum = sum(
                [conf_matrix[i][i] for i in range(len(conf_matrix))])
            norm_diagonal_sum = sum(
                [float(conf_matrix[i][i]) / float(sum(conf_matrix[i])) for i in
                 range(len(conf_matrix))])
            total_count = np.sum(conf_matrix)
            correct = 0
            for i in range(len(conf_matrix)):
                correct += conf_matrix[i][i] + conf_matrix[i][max(i - 1, 0)] * ((i - 1) >= 0) + \
                           conf_matrix[i][min(i + 1, len(conf_matrix[i]) - 1)] * ((i + 1) <= len(conf_matrix[i]) - 1)
            # print 'Accuracy [-1, +1]:', float(correct) / float(total_count)
            print 'Accuracy:', float(diagonal_sum) / float(total_count)
            print 'Balanced accuracy:', float(norm_diagonal_sum) / float(conf_matrix.shape[0])
            avg_nodes['inTrees'].append(len(orl.rule_list))

            print 'GENESIM'
            # train_gen = train.rename(columns={'Class': 'cat'})
            start = time.time()
            genetic = genesim.genetic_algorithm(train, label_col, _constructors, seed=None, num_iterations=25,
                                               num_crossovers=10, population_size=150, val_fraction=0.5, prune=True,
                                               max_samples=1, tournament_size=10, nr_bootstraps=25)
            end = time.time()
            times['GENESIM'].append(end - start)
            predictions = genetic.evaluate_multiple(X_test).astype(int)
            conf_matrices['GENESIM'].append(confusion_matrix(y_test, predictions))
            conf_matrix = confusion_matrix(y_test, predictions)
            print conf_matrix
            diagonal_sum = sum(
                [conf_matrix[i][i] for i in range(len(conf_matrix))])
            norm_diagonal_sum = sum(
                [float(conf_matrix[i][i]) / float(sum(conf_matrix[i])) for i in
                 range(len(conf_matrix))])
            total_count = np.sum(conf_matrix)
            correct = 0
            for i in range(len(conf_matrix)):
                correct += conf_matrix[i][i] + conf_matrix[i][max(i - 1, 0)] * ((i - 1) >= 0) + \
                           conf_matrix[i][min(i + 1, len(conf_matrix[i]) - 1)] * ((i + 1) <= len(conf_matrix[i]) - 1)
            # print 'Accuracy [-1, +1]:', float(correct) / float(total_count)
            print 'Accuracy:', float(diagonal_sum) / float(total_count)
            print 'Balanced accuracy:', float(norm_diagonal_sum) / float(conf_matrix.shape[0])
            avg_nodes['GENESIM'].append(genetic.count_nodes())

        print times
        print avg_nodes

        fig = plt.figure()
        fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(NR_FOLDS) + ' folds', fontsize=20)
        counter = 0
        conf_matrices_mean = {}
        for key in conf_matrices:
            conf_matrices_mean[key] = np.zeros(conf_matrices[key][0].shape)
            for i in range(len(conf_matrices[key])):
                conf_matrices_mean[key] = np.add(conf_matrices_mean[key], conf_matrices[key][i])
            cm_normalized = np.around(
                conf_matrices_mean[key].astype('float') / conf_matrices_mean[key].sum(axis=1)[:,
                                                          np.newaxis], 4)

            diagonal_sum = sum(
                [conf_matrices_mean[key][i][i] for i in range(len(conf_matrices_mean[key]))])
            norm_diagonal_sum = sum(
                [conf_matrices_mean[key][i][i]/sum(conf_matrices_mean[key][i]) for i in range(len(conf_matrices_mean[key]))])
            total_count = np.sum(conf_matrices_mean[key])
            print key
            print conf_matrices_mean[key]
            correct = 0
            for i in range(len(conf_matrices_mean[key])):
                correct += conf_matrices_mean[key][i][i] + conf_matrices_mean[key][i][max(i - 1, 0)] * ((i - 1) >= 0) + \
                           conf_matrices_mean[key][i][min(i + 1, len(conf_matrices_mean[key][i]) - 1)] * ((i + 1) <= len(conf_matrices_mean[key][i]) - 1)
            print 'Accuracy [-1, +1]:', float(correct) / float(total_count)
            print 'Accuracy:', float(diagonal_sum) / float(total_count)
            print 'Balanced accuracy:', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]

            ax = fig.add_subplot(2, np.math.ceil(len(conf_matrices) / 2.0), counter + 1)
            cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
            ax.set_title(key + '(' + str(sum(avg_nodes[key])/len(avg_nodes[key])) + ')', y=1.08)
            for (j, i), label in np.ndenumerate(cm_normalized):
                ax.text(i, j, label, ha='center', va='center')
            if counter == len(conf_matrices) - 1:
                fig.colorbar(cax, fraction=0.046, pad=0.04)
            counter += 1
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0] * 2, Size[1] * 1.75, forward=True)
        plt.show()
        rand_nr = str(int(10000*np.random.rand()))
        plt.savefig('output/' + dataset['name'] + '_CV'+str(NR_FOLDS)+'.png', bbox_inches='tight')