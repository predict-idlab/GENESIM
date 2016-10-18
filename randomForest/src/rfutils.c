/*******************************************************************
   Copyright (C) 2001-2012 Leo Breiman, Adele Cutler and Merck & Co., Inc.

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
#include "rf.h"


void zeroInt(int *x, int length) {
    memset(x, 0, length * sizeof(int));
}

void zeroDouble(double *x, int length) {
    memset(x, 0, length * sizeof(double));
}

void createClass(double *x, int realN, int totalN, int mdim) {
/* Create the second class by bootstrapping each variable independently. */
    int i, j, k;
    for (i = realN; i < totalN; ++i) {
        for (j = 0; j < mdim; ++j) {
            k = (int) (unif_rand() * realN);
            x[j + i * mdim] = x[j + k * mdim];
        }
    }
}

void normClassWt(int *cl, const int nsample, const int nclass,
                 const int useWt, double *classwt, int *classFreq) {
    int i;
    double sumwt = 0.0;

    if (useWt) {
        /* Normalize user-supplied weights so they sum to one. */
        for (i = 0; i < nclass; ++i) sumwt += classwt[i];
        for (i = 0; i < nclass; ++i) classwt[i] /= sumwt;
    } else {
        for (i = 0; i < nclass; ++i) {
            classwt[i] = ((double) classFreq[i]) / nsample;
        }
    }
    for (i = 0; i < nclass; ++i) {
        classwt[i] = classFreq[i] ? classwt[i] * nsample / classFreq[i] : 0.0;
    }
}

void makeA(double *x, const int mdim, const int nsample, int *cat, int *a,
           int *b) {
    /* makeA() constructs the mdim by nsample integer array a.  For each
       numerical variable with values x(m, n), n=1, ...,nsample, the x-values
       are sorted from lowest to highest.  Denote these by xs(m, n).  Then
       a(m,n) is the case number in which xs(m, n) occurs. The b matrix is
       also contructed here.  If the mth variable is categorical, then
       a(m, n) is the category of the nth case number. */
    int i, j, n1, n2, *index;
    double *v;

    v     = (double *) Calloc(nsample, double);
    index = (int *) Calloc(nsample, int);

    for (i = 0; i < mdim; ++i) {
        if (cat[i] == 1) { /* numerical predictor */
            for (j = 0; j < nsample; ++j) {
                v[j] = x[i + j * mdim];
                index[j] = j + 1;
            }
            R_qsort_I(v, index, 1, nsample);

            /*  this sorts the v(n) in ascending order. index(n) is the case
                number of that v(n) nth from the lowest (assume the original
                case numbers are 1,2,...).  */
            for (j = 0; j < nsample-1; ++j) {
                n1 = index[j];
                n2 = index[j + 1];
                a[i + j * mdim] = n1;
                if (j == 0) b[i + (n1-1) * mdim] = 1;
                b[i + (n2-1) * mdim] =  (v[j] < v[j + 1]) ?
                    b[i + (n1-1) * mdim] + 1 : b[i + (n1-1) * mdim];
            }
            a[i + (nsample-1) * mdim] = index[nsample-1];
        } else { /* categorical predictor */
            for (j = 0; j < nsample; ++j)
                a[i + j*mdim] = (int) x[i + j * mdim];
        }
    }
    Free(index);
    Free(v);
}


void modA(int *a, int *nuse, const int nsample, const int mdim,
	  int *cat, const int maxcat, int *ncase, int *jin) {
    int i, j, k, m, nt;

    *nuse = 0;
    for (i = 0; i < nsample; ++i) if (jin[i]) (*nuse)++;

    for (i = 0; i < mdim; ++i) {
      k = 0;
      nt = 0;
      if (cat[i] == 1) {
          for (j = 0; j < nsample; ++j) {
              if (jin[a[i + k * mdim] - 1]) {
                  a[i + nt * mdim] = a[i + k * mdim];
                  k++;
              } else {
                  for (m = 0; m < nsample - k; ++m) {
                      if (jin[a[i + (k + m) * mdim] - 1]) {
                          a[i + nt * mdim] = a[i + (k + m) * mdim];
                          k += m + 1;
                          break;
                      }
                  }
              }
              nt++;
              if (nt >= *nuse) break;
          }
      }
    }
    if (maxcat > 1) {
        k = 0;
        nt = 0;
        for (i = 0; i < nsample; ++i) {
            if (jin[k]) {
                k++;
                ncase[nt] = k;
            } else {
                for (j = 0; j < nsample - k; ++j) {
                    if (jin[k + j]) {
                        ncase[nt] = k + j + 1;
                        k += j + 1;
                        break;
                    }
                }
            }
            nt++;
            if (nt >= *nuse) break;
        }
    }
}

void Xtranslate(double *x, int mdim, int nrnodes, int nsample,
		int *bestvar, int *bestsplit, int *bestsplitnext,
		double *xbestsplit, int *nodestatus, int *cat, int treeSize) {
/*
 this subroutine takes the splits on numerical variables and translates them
 back into x-values.  It also unpacks each categorical split into a
 32-dimensional vector with components of zero or one--a one indicates that
 the corresponding category goes left in the split.
*/

    int i, m;

    for (i = 0; i < treeSize; ++i) {
	if (nodestatus[i] == 1) {
	    m = bestvar[i] - 1;
	    if (cat[m] == 1) {
		xbestsplit[i] = 0.5 * (x[m + (bestsplit[i] - 1) * mdim] +
				       x[m + (bestsplitnext[i] - 1) * mdim]);
	    } else {
		xbestsplit[i] = (double) bestsplit[i];
	    }
	}
    }
}

void permuteOOB(int m, double *x, int *in, int nsample, int mdim) {
/* Permute the OOB part of a variable in x.
 * Argument:
 *   m: the variable to be permuted
 *   x: the data matrix (variables in rows)
 *   in: vector indicating which case is OOB
 *   nsample: number of cases in the data
 *   mdim: number of variables in the data
 */
    double *tp, tmp;
    int i, last, k, nOOB = 0;

    tp = (double *) Calloc(nsample, double);

    for (i = 0; i < nsample; ++i) {
		/* make a copy of the OOB part of the data into tp (for permuting) */
		if (in[i] == 0) {
            tp[nOOB] = x[m + i*mdim];
            nOOB++;
        }
    }
    /* Permute tp */
    last = nOOB;
    for (i = 0; i < nOOB; ++i) {
		k = (int) last * unif_rand();
		tmp = tp[last - 1];
		tp[last - 1] = tp[k];
		tp[k] = tmp;
		last--;
    }

    /* Copy the permuted OOB data back into x. */
    nOOB = 0;
    for (i = 0; i < nsample; ++i) {
		if (in[i] == 0) {
            x[m + i*mdim] = tp[nOOB];
            nOOB++;
		}
    }
    Free(tp);
}

/* Compute proximity. */
void computeProximity(double *prox, int oobprox, int *node, int *inbag,
                      int *oobpair, int n) {
/* Accumulate the number of times a pair of points fall in the same node.
   prox:    n x n proximity matrix
   oobprox: should the accumulation only count OOB cases? (0=no, 1=yes)
   node:    vector of terminal node labels
   inbag:   indicator of whether a case is in-bag
   oobpair: matrix to accumulate the number of times a pair is OOB together
   n:       total number of cases
*/
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = i+1; j < n; ++j) {
            if (oobprox) {
                if (! (inbag[i] > 0 || inbag[j] > 0) ) {
                    oobpair[j*n + i] ++;
                    oobpair[i*n + j] ++;
                    if (node[i] == node[j]) {
                        prox[j*n + i] += 1.0;
                        prox[i*n + j] += 1.0;
                    }
                }
            } else {
                if (node[i] == node[j]) {
                    prox[j*n + i] += 1.0;
                    prox[i*n + j] += 1.0;
                }
            }
        }
    }
}

double pack(const int nBits, const int *bits) {
    int i = nBits - 1;
	double pack = bits[i];
    for (i = nBits - 1; i > 0; --i) pack = 2.0 * pack + bits[i - 1];
    return(pack);
}

void unpack(const double pack, const int nBits, int *bits) {
    int i;
    double x = pack;
    for (i = 0; i <= nBits; ++i) {
    	bits[i] = ((unsigned long) x & 1) ? 1 : 0;
    	x = x / 2;
    }
}

/*
unsigned int pack(int nBits, int *bits) {
    int i = nBits;
	unsigned int pack = 0;
    while (--i >= 0) pack += bits[i] << i;
    return(pack);
}

void unpack(int nBits, unsigned int pack, int *bits) {
    int i;
    for (i = 0; i < nBits; pack >>= 1, ++i) bits[i] = pack & 1;
}
*/

void F77_NAME(unpack)(double *pack, int *nBits, int *bits) {
	unpack(*pack, *nBits, bits);
}

