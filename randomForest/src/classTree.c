/*******************************************************************
   Copyright (C) 2001-9 Leo Breiman, Adele Cutler and Merck & Co., Inc.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
*******************************************************************/

#include <R.h>
#include <Rmath.h>
#include "rf.h"

#ifdef C_CLASSTREE
void classTree(int *a, int *b, int *class, int *cat, int mdim, int nsample,
               int nclass, int *treemap, int *bestvar, double *bestsplit,
               double *bestsplitnext, double *tgini, int *nodeStatus,
               int *nodePop, int *nodeStart, double *tclassPop, int maxNodes,
               int nodeSize, int *ncase, int *inBag, int mTry, int *varUsed,
               int *nodeClass, int *treeSize, double *win) {
/* Buildtree consists of repeated calls to two subroutines, Findbestsplit
   and Movedata.  Findbestsplit does just that--it finds the best split of
   the current node.  Movedata moves the data in the split node right and
   left so that the data corresponding to each child node is contiguous.
   The buildtree bookkeeping is different from that in Friedman's original
   CART program.  ncur is the total number of nodes to date.
   nodeStatus(k)=1 if the kth node has been split.  nodeStatus(k)=2 if the
   node exists but has not yet been split, and =-1 of the node is terminal.
   A node is terminal if its size is below a threshold value, or if it is
   all one class, or if all the x-values are equal.  If the current node k
   is split, then its children are numbered ncur+1 (left), and
   ncur+2(right), ncur increases to ncur+2 and the next node to be split is
   numbered k+1.  When no more nodes can be split, buildtree returns to the
   main program.
*/
/*
  integer a(mdim,nsample),cl(nsample),cat(mdim),
  treemap(2,numNodes),bestvar(numNodes),
          bestsplit(numNodes), nodeStatus(numNodes),ta(nsample),
          nodePop(numNodes),nodeStart(numNodes),
          bestsplitnext(numNodes),idmove(nsample),
          ncase(nsample),parent(numNodes),b(mdim,nsample),
          jin(nsample),iv(mred),nodeclass(numNodes),mind(mred)


      double precision tclasspop(nclass),classpop(nclass,numNodes),
     1     tclasscat(nclass,MAX_CAT),win(nsample),wr(nclass),wc(nclass),
     1     wl(nclass),tgini(mdim), xrand
 */
    int msplit = 0, i, j;
    zeroInt(nodeStatus, maxNodes);
    zeroInt(nodeStart, maxNodes);
    zeroInt(nodePop, maxNodes);
    zeroDouble(classPop, nclass * maxNodes);

    for (i = 0; i < nclass; ++i) classPop[i] = tclassPop[i];
    ncur = 1;
    nodeStart[0] = 1;
    nodePop[0] = *nuse;
    nodeStatus[0] = NODE_TOSPLIT;
    /* 2: not split yet, 1: split, -1: terminal */
    /* start main loop */
    for (i = 0; i < numNodes; ++i) {
        if (i > ncur - 1) break;
        if (nodeStatus[i] != NODE_TOSPLIT) continue;
        /* initialize for next call to findbestsplit */
        ndstart = nodeStart[i];
        ndend = ndstart + nodePop[i] - 1;
        for (j = 0; j < nclass; ++j) {
            tclassPop[j] = classPop[j + i * nclass];
        }
        jstat = 0;
        F77_CALL(findbestsplit)(a, b, cl, mdim, nsample, nclass, cat,
                                ndstart, ndend, tclassPop, tclasscat,
                                &msplit, &decsplit, &nbest, ncase, &jstat,
                                inBag, mTry, win, wr, wc, wl, mred, i, mind);
        if (jstat == 1) {
            nodeStatus[i] = NODE_TERMINAL;
            continue;
        } else {
            bestvar[i] = msplit;
            varUsed[msplit - 1] = 1;
            tgini[msplit - 1] += decsplit;
            if (cat[msplit-1] == 1) {
                bestsplit[i] = a[msplit - 1  + nbest * mdim];
                bestsplitnext[i] = a[msplit - 1 + (nbest + 1) * mdim];
            } else {
                  bestsplit[i] = nbest;
                  bestsplitnext[i] = 0;
            }
        }
        F77_CALL(movedata)(a, ta, mdim, nsample, ndstart, ndend, idmove,
                           ncase, msplit, cat, nbest, ndendl);
        /* leftnode no.= ncur+1, rightnode no. = ncur+2. */
        nodePop[ncur+1] = ndendl - ndstart + 1;
        nodePop[ncur+2] = ndend - ndendl;
        nodeStart[ncur+1] = ndstart;
        nodeStart[ncur+2] = ndendl + 1;
        /* find class populations in both nodes */
        for (n = ndstart; n <= ndendl; ++n) {
            nc = ncase[n];
            j = class[nc-1];
            classPop[j - 1 + (ncur+1)*mdim] += win[nc - 1];
        }
        for (n = ndendl + 1; n <= ndend; ++n) {
            nc = ncase[n];
            j = cl[nc - 1];
            classPop[j - 1 + (ncur+2) * mdim] += win[nc - 1];
        }
        /* check on nodeStatus */
        nodeStatus[ncur + 1] = NODE_TOSPLIT;
        nodeStatus[ncur + 2] = NODE_TOSPLIT;
        if (nodePop[ncur + 1] <= ndsize) nodeStatus[ncur+1] = NODE_TERMINAL;
        if (nodePop[ncur + 2] <= ndsize) nodeStatus[ncur+2] = NODE_TERMINAL;
        popt1 = 0;
        popt2 = 0;
        for (j = 0; j < nclass; ++j) {
            popt1 += classPop[j + (ncur+1) * mdim];
            popt2 += classPop[j + (ncur+2) * mdim];
        }
        for (j = 0; j < nclass; ++j) {
            if (classPop[j + (ncur+1) * mdim] == popt1)
                nodeStatus[ncur+1] = NODE_TERMINAL;
            if (classPop[j + (ncur+2) * mdim] == popt2)
                nodeStatus[ncur+2] = NODE_TERMINAL;
        }

        treemap[i * 2] = ncur + 1;
        treemap[1 + i * 2] = ncur + 2;
        nodeStatus[i] = NODE_INTERIOR;
        ncur += 2;
        if (ncur >= numNodes) break;
    }
    ndbigtree = numNodes;
    for (k = numNodes-1; k >= 0; --k) {
        if (nodeStatus[k] == 0) ndbigtree--;
        if (nodeStatus[k] == NODE_TOSPLIT) nodeStatus[k] = NODE_TERMINAL;
    }
    for (k = 0; k < ndbigtree; ++k) {
        if (nodeStatus[k] == NODE_TERMINAL) {
            pp = 0;
            ntie = 1;
            for (j = 0; j < nclass; ++j) {
                if (classPop[j + k * nclass] > pp) {
                    nodeClass[k] = j;
                    pp = classPop[j + k * nclass];
                    ntie = 1;
                }
                /* Break ties at random: */
                if (classPop[j + k * nclass] == pp) {
                	if (unif_rand() < 1.0 / ntie) {
                		nodeClass[k] = j;
                		pp = classPop[j + k * nclass];
                	}
                	ntie++;
                }
            }
        }
    }
}



void findBestSplit(int *a, double *b, int *class, int mDim, int nSample,
                   int nClass, int *nCat, int maxCat, int ndStart, int ndEnd,
                   double *classCount, double *classCatTable,
                   int *splitVar, double *decGini, int *bestSplit,
                   int *ncase, int *splitStatus, int *inBag, int mtry,
                   double *weight, double *wr, double *wc, double *wl,
                   int *currentNode, int *mind) {
/*
      subroutine findbestsplit(a, b, cl, mdim, nsample, nclass, cat,
     1     maxcat, ndstart, ndend, tclasspop, tclasscat, msplit,
     2     decsplit, nbest, ncase, jstat, jin, mtry, win, wr, wc, wl,
     3     mred, kbuild, mind) */
/*
     For the best split, msplit is the variable split on. decsplit is the
     dec. in impurity.  If msplit is numerical, nsplit is the case number
     of value of msplit split on, and nsplitnext is the case number of the
     next larger value of msplit.  If msplit is categorical, then nsplit is
     the coding into an integer of the categories going left.
*/


    integer a(mdim,nsample), cl(nsample), cat(mdim),
     1     ncase(nsample), b(mdim,nsample), jin(nsample), nn, j
      double precision tclasspop(nclass), tclasscat(nclass,MAX_CAT), dn(MAX_CAT),
     1     win(nsample), wr(nclass), wc(nclass), wl(nclass), xrand
      integer mind(mred), ncmax, ncsplit,nhit
        ncmax = 10;
    ncsplit = 512;
    /* compute initial values of numerator and denominator of Gini */
    parentNum = 0.0;
    parentDen = 0.0;
    for (i = 0; i < nClass; ++i) {
        parentNum += classCount[i] * classCount[i];
        parentDen += classCount[i];
    }
    crit0 = pno / pdo;
    *splitStatus = 0;
    critmax = -1.0e25;
    for (i = 0; i < mDim; ++i) mind[i] = i;

    /* start main loop through variables to find best split. */
    last = mDim - 1;
    for (i = 0, i < mtry; ++i) {
        /* sample mtry variables w/o replacement. */
        j = (int) (unif_rand() * (last + 1));
        mvar = mIndex[j];
        swapInt(mIndex[j], mIndex[last]);
        last--;

        lcat = nCat[mvar];
        if (lcat == 1) {
            /* Split on a numerical predictor. */
            rightNum = parentNum;
            rightDen = parentDen;
            leftNum = 0.0;
            leftDen = 0.0;
            zeroDouble(wl, nClass);
            for (j = 0; j < nClass; ++j) wr[j] = classCount[j];
	    ntie = 1;
            for (j = ndstart; j <= ndend - 1; ++j) {
                nc = a[mvar, j-1];
                u = weight[nc];
                k = class[nc];
                leftNum += u * (2 * wl[k-1] + u);
                rightNum += u * (-2 * wr[k-1] + u);
                leftDen += u;
                rightDen -= u;
                wl[k-1] += u;
                wr[k-1] -= u;
                if (b[mvar, nc] < b[mvar, a[mvar, j]]) {
                    if (fmin2(rightDen, leftDen) > 1.0e-5) {
                        crit = (leftNum / leftDen) + (rightNum / rightDen);
                        if (crit > critmax) {
                            *bestSplit = j;
                            critmax = crit;
                            *splitVar = mvar;
                            ntie = 1;
                        }
                        /* Break ties at random: */
                        if (crit == critmax) {
                        	if (unif_rand() < 1.0 / ntie) {
                        		*bestSplit = j;
                        		critmax = crit;
                        		*splitVar = mvar;
                        	}
                        	ntie++;
                        }
                    }
                }
            }
        } else {
            /* Split on a categorical predictor. */
            zeroDouble(classCatTable, nClass * MAX_CAT);
            for (j = ndstart; j <= ndend; ++j) {
                nc = ncase[j-1];
                l = a[mvar, ncase[j-1]];
                classCatTable[class[nc-1], l-1] += weight[nc-1];
            }
            nNotEmpty = 0;
            for (j = 0; j < lcat; ++j) {
                catSum = 0;
                for (k = 0; k < nClass; ++k) {
                    catSum += classCatTable[k, j];
                }
                catCount[j] = su;
                if (catSum > 0) nNotEmpty ++;
            }
            nhit = 0;
            if (nNotEmpty > 1) {
                F77_CALL(catmax)(parentden, classcatTable, classCount,
                                 &nclass, &lcat, bestSplit, &critmax, &nhit,
                                 &maxcat, &ncmax, &ncsplit);
            }
            if (nhit) *splitVar = mvar;
        }
    }
    if (critmax < -1.0e10 || msplit == 0) {
        *splitStatus = -1;
    } else {
        *decsplit = critmax - crit0;
    }
}
#endif /* C_CLASSTREE */



void F77_NAME(catmax)(double *parentDen, double *tclasscat,
                      double *tclasspop, int *nclass, int *lcat,
                      double *catsp, double *critmax, int *nhit,
                      int *maxcat, int *ncmax, int *ncsplit) {
/* This finds the best split of a categorical variable with lcat
   categories and nclass classes, where tclasscat(j, k) is the number
   of cases in class j with category value k. The method uses an
   exhaustive search over all partitions of the category values if the
   number of categories is 10 or fewer.  Otherwise ncsplit randomly
   selected splits are tested and best used. */
    int j, k, n, icat[MAX_CAT], nsplit;
    double leftNum, leftDen, rightNum, decGini, *leftCatClassCount;

    leftCatClassCount = (double *) Calloc(*nclass, double);
    *nhit = 0;
    nsplit = *lcat > *ncmax ?
        *ncsplit : (int) pow(2.0, (double) *lcat - 1) - 1;

    for (n = 0; n < nsplit; ++n) {
        zeroInt(icat, MAX_CAT);
        if (*lcat > *ncmax) {
            /* Generate random split.
               TODO: consider changing to generating random bits with more
               efficient algorithm */
            for (j = 0; j < *lcat; ++j) icat[j] = unif_rand() > 0.5 ? 1 : 0;
        } else {
            unpack((double) n + 1, *lcat, icat);
        }
        for (j = 0; j < *nclass; ++j) {
            leftCatClassCount[j] = 0;
            for (k = 0; k < *lcat; ++k) {
                if (icat[k]) {
                    leftCatClassCount[j] += tclasscat[j + k * *nclass];
                }
            }
        }
        leftNum = 0.0;
        leftDen = 0.0;
        for (j = 0; j < *nclass; ++j) {
            leftNum += leftCatClassCount[j] * leftCatClassCount[j];
            leftDen += leftCatClassCount[j];
        }
        /* If either node is empty, try another split. */
        if (leftDen <= 1.0e-8 || *parentDen - leftDen <= 1.0e-5) continue;
        rightNum = 0.0;
        for (j = 0; j < *nclass; ++j) {
            leftCatClassCount[j] = tclasspop[j] - leftCatClassCount[j];
            rightNum += leftCatClassCount[j] * leftCatClassCount[j];
        }
        decGini = (leftNum / leftDen) + (rightNum / (*parentDen - leftDen));
        if (decGini > *critmax) {
            *critmax = decGini;
            *nhit = 1;
            *catsp = *lcat > *ncmax ? pack(*lcat, icat) : n + 1;
        }
    }
    Free(leftCatClassCount);
}



/* Find best split of with categorical variable when there are two classes */
void F77_NAME(catmaxb)(double *totalWt, double *tclasscat, double *classCount,
                       int *nclass, int *nCat, double *best, double *critmax,
                       int *nhit, double *catCount) {

    double catProportion[MAX_CAT], cp[MAX_CAT], cm[MAX_CAT];
    int kcat[MAX_CAT];
    int i, j;
    double bestsplit=0.0, rightDen, leftDen, leftNum, rightNum, crit;

    *nhit = 0;
    for (i = 0; i < *nCat; ++i) {
        catProportion[i] = catCount[i] ?
            tclasscat[i * *nclass] / catCount[i] : 0.0;
        kcat[i] = i + 1;
    }
    R_qsort_I(catProportion, kcat, 1, *nCat);
    for (i = 0; i < *nclass; ++i) {
        cp[i] = 0;
        cm[i] = classCount[i];
    }
    rightDen = *totalWt;
    leftDen = 0.0;
    for (i = 0; i < *nCat - 1; ++i) {
        leftDen += catCount[kcat[i]-1];
        rightDen -= catCount[kcat[i]-1];
        leftNum = 0.0;
        rightNum = 0.0;
        for (j = 0; j < *nclass; ++j) {
            cp[j] += tclasscat[j + (kcat[i]-1) * *nclass];
            cm[j] -= tclasscat[j + (kcat[i]-1) * *nclass];
            leftNum += cp[j] * cp[j];
            rightNum += cm[j] * cm[j];
        }
        if (catProportion[i] < catProportion[i + 1]) {
            /* If neither node is empty, check the split. */
            if (rightDen > 1.0e-5 && leftDen > 1.0e-5) {
                crit = (leftNum / leftDen) + (rightNum / rightDen);
                if (crit > *critmax) {
                    *critmax = crit;
                    bestsplit = .5 * (catProportion[i] + catProportion[i + 1]);
                    *nhit = 1;
                }
            }
        }
    }
    if (*nhit == 1) {
        zeroInt(kcat, *nCat);
        for (i = 0; i < *nCat; ++i) {
            catProportion[i] = catCount[i] ?
                tclasscat[i * *nclass] / catCount[i] : 0.0;
            kcat[i] = catProportion[i] < bestsplit ? 1 : 0;
			/* Rprintf("%i ", kcat[i]); */
        }
        *best = pack(*nCat, kcat);
		/* Rprintf("\nnbest=%u\nnbest=%i\n", *nbest, *nbest); */
    }
}



void predictClassTree(double *x, int n, int mdim, int *treemap,
		      int *nodestatus, double *xbestsplit,
		      int *bestvar, int *nodeclass,
		      int treeSize, int *cat, int nclass,
		      int *jts, int *nodex, int maxcat) {
    int m, i, j, k, *cbestsplit;
	double dpack;

    /* decode the categorical splits */
    if (maxcat > 1) {
        cbestsplit = (int *) Calloc(maxcat * treeSize, int);
        zeroInt(cbestsplit, maxcat * treeSize);
        for (i = 0; i < treeSize; ++i) {
            if (nodestatus[i] != NODE_TERMINAL) {
                if (cat[bestvar[i] - 1] > 1) {
                    dpack = xbestsplit[i];
                    /* unpack `dpack' into bits */
                    /* unpack(dpack, maxcat, cbestsplit + i * maxcat); */
                    for (j = 0; j < cat[bestvar[i] - 1]; ++j) {
                    	cbestsplit[j + i*maxcat] = ((unsigned long) dpack & 1) ? 1 : 0;
                    	dpack = dpack / 2;
                    }
                }
            }
        }
    }
    for (i = 0; i < n; ++i) {
		k = 0;
		while (nodestatus[k] != NODE_TERMINAL) {
            m = bestvar[k] - 1;
            if (cat[m] == 1) {
				/* Split by a numerical predictor */
				k = (x[m + i * mdim] <= xbestsplit[k]) ?
					treemap[k * 2] - 1 : treemap[1 + k * 2] - 1;
			} else {
				/* Split by a categorical predictor */
				k = cbestsplit[(int) x[m + i * mdim] - 1 + k * maxcat] ?
					treemap[k * 2] - 1 : treemap[1 + k * 2] - 1;
			}
		}
		/* Terminal node: assign class label */
		jts[i] = nodeclass[k];
		nodex[i] = k + 1;
    }
    if (maxcat > 1) Free(cbestsplit);
}
