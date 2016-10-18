"""
This is an example script that will apply k-cross-validation on all datasets with a load function in
`data.load_datasets` and for all implemented tree constructors, ensemble techniques and GENESIM. In the end,
a confusion matrices will be stored at path `output/dataset_name_CVk.png` and the average model complexity and
computational time required for each of the algorithms will be printed out.

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""


import time

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold

import matplotlib.pyplot as plt
import numpy as np

from constructors.ensemble import RFClassification, XGBClassification
from constructors.genesim import GENESIM
from constructors.treeconstructor import QUESTConstructor, GUIDEConstructor, C45Constructor, CARTConstructor
from data.load_all_datasets import load_all_datasets
from decisiontree import DecisionTree

if __name__ == "__main__":

    algorithms = {QUESTConstructor().get_name(): QUESTConstructor(), GUIDEConstructor().get_name(): GUIDEConstructor(),
                  CARTConstructor().get_name(): CARTConstructor(), C45Constructor().get_name(): C45Constructor(),
                  RFClassification().get_name(): RFClassification(), XGBClassification().get_name(): XGBClassification()}
    genesim = GENESIM()

    NR_FOLDS = 3
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

        skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=None)

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
                conf_matrices[algorithm].append(confusion_matrix(y_test, predictions))
                if type(clf) is DecisionTree:
                    avg_nodes[algorithm].append(clf.count_nodes())
                else:
                    avg_nodes[algorithm].append(clf.nr_clf)

            print 'GENESIM'
            # train_gen = train.rename(columns={'Class': 'cat'})
            start = time.time()
            constructors = [C45Constructor(), CARTConstructor(), QUESTConstructor(), GUIDEConstructor()]
            genetic = genesim.genetic_algorithm(train, label_col, constructors, seed=None, num_iterations=15,
                                               num_crossovers=10, population_size=150, val_fraction=0.5, prune=True,
                                               max_samples=1, tournament_size=10, nr_bootstraps=25)
            end = time.time()
            times['GENESIM'].append(end - start)
            predictions = genetic.evaluate_multiple(X_test).astype(int)
            conf_matrices['GENESIM'].append(confusion_matrix(y_test, predictions))
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
        # plt.show()
        rand_nr = str(int(10000*np.random.rand()))
        plt.savefig('output/' + dataset['name'] + '_CV'+str(NR_FOLDS)+'.png', bbox_inches='tight')