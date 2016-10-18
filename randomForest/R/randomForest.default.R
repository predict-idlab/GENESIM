## mylevels() returns levels if given a factor, otherwise 0.
mylevels <- function(x) if (is.factor(x)) levels(x) else 0

"randomForest.default" <-
    function(x, y=NULL,  xtest=NULL, ytest=NULL, ntree=500,
             mtry=if (!is.null(y) && !is.factor(y))
             max(floor(ncol(x)/3), 1) else floor(sqrt(ncol(x))),
             replace=TRUE, classwt=NULL, cutoff, strata,
             sampsize = if (replace) nrow(x) else ceiling(.632*nrow(x)),
             nodesize = if (!is.null(y) && !is.factor(y)) 5 else 1,
             maxnodes=NULL,
             importance=FALSE, localImp=FALSE, nPerm=1,
             proximity, oob.prox=proximity,
             norm.votes=TRUE, do.trace=FALSE,
             keep.forest=!is.null(y) && is.null(xtest), corr.bias=FALSE,
             keep.inbag=FALSE, ...) {
    addclass <- is.null(y)
    classRF <- addclass || is.factor(y)
    if (!classRF && length(unique(y)) <= 5) {
        warning("The response has five or fewer unique values.  Are you sure you want to do regression?")
    }
    if (classRF && !addclass && length(unique(y)) < 2)
        stop("Need at least two classes to do classification.")
    n <- nrow(x)
    p <- ncol(x)
    if (n == 0) stop("data (x) has 0 rows")
    x.row.names <- rownames(x)
    x.col.names <- if (is.null(colnames(x))) 1:ncol(x) else colnames(x)

    ## overcome R's lazy evaluation:
    keep.forest <- keep.forest

    testdat <- !is.null(xtest)
    if (testdat) {
        if (ncol(x) != ncol(xtest))
            stop("x and xtest must have same number of columns")
        ntest <- nrow(xtest)
        xts.row.names <- rownames(xtest)
    }

    ## Make sure mtry is in reasonable range.
    if (mtry < 1 || mtry > p)
        warning("invalid mtry: reset to within valid range")
    mtry <- max(1, min(p, round(mtry)))
    if (!is.null(y)) {
        if (length(y) != n) stop("length of response must be the same as predictors")
        addclass <- FALSE
    } else {
        if (!addclass) addclass <- TRUE
        y <- factor(c(rep(1, n), rep(2, n)))
        x <- rbind(x, x)
    }

    ## Check for NAs.
    if (any(is.na(x))) stop("NA not permitted in predictors")
    if (testdat && any(is.na(xtest))) stop("NA not permitted in xtest")
    if (any(is.na(y))) stop("NA not permitted in response")
    if (!is.null(ytest) && any(is.na(ytest))) stop("NA not permitted in ytest")

    if (is.data.frame(x)) {
        xlevels <- lapply(x, mylevels)
        ncat <- sapply(xlevels, length)
        ## Treat ordered factors as numerics.
        ncat <- ifelse(sapply(x, is.ordered), 1, ncat)
        x <- data.matrix(x)
        if(testdat) {
            if(!is.data.frame(xtest))
                stop("xtest must be data frame if x is")
            xfactor <- which(sapply(xtest, is.factor))
            if (length(xfactor) > 0) {
                for (i in xfactor) {
                    if (any(! levels(xtest[[i]]) %in% xlevels[[i]]))
                        stop("New factor levels in xtest not present in x")
                    xtest[[i]] <-
                        factor(xlevels[[i]][match(xtest[[i]], xlevels[[i]])],
                               levels=xlevels[[i]])
                }
            }
            xtest <- data.matrix(xtest)
        }
    } else {
        ncat <- rep(1, p)
		names(ncat) <- colnames(x)
        xlevels <- as.list(rep(0, p))
    }
    maxcat <- max(ncat)
    if (maxcat > 53)
        stop("Can not handle categorical predictors with more than 53 categories.")

    if (classRF) {
        nclass <- length(levels(y))
        ## Check for empty classes:
        if (any(table(y) == 0)) stop("Can't have empty classes in y.")
        if (!is.null(ytest)) {
            if (!is.factor(ytest)) stop("ytest must be a factor")
            if (!all(levels(y) == levels(ytest)))
                stop("y and ytest must have the same levels")
        }
        if (missing(cutoff)) {
            cutoff <- rep(1 / nclass, nclass)
        } else {
            if (sum(cutoff) > 1 || sum(cutoff) < 0 || !all(cutoff > 0) ||
                length(cutoff) != nclass) {
                stop("Incorrect cutoff specified.")
            }
            if (!is.null(names(cutoff))) {
                if (!all(names(cutoff) %in% levels(y))) {
                    stop("Wrong name(s) for cutoff")
                }
                cutoff <- cutoff[levels(y)]
            }
        }
        if (!is.null(classwt)) {
            if (length(classwt) != nclass)
                stop("length of classwt not equal to number of classes")
            ## If classwt has names, match to class labels.
            if (!is.null(names(classwt))) {
                if (!all(names(classwt) %in% levels(y))) {
                    stop("Wrong name(s) for classwt")
                }
                classwt <- classwt[levels(y)]
            }
            if (any(classwt <= 0)) stop("classwt must be positive")
            ipi <- 1
        } else {
            classwt <- rep(1, nclass)
            ipi <- 0
        }
    } else addclass <- FALSE

    if (missing(proximity)) proximity <- addclass
    if (proximity) {
        prox <- matrix(0.0, n, n)
        proxts <- if (testdat) matrix(0, ntest, ntest + n) else double(1)
    } else {
        prox <- proxts <- double(1)
    }

    if (localImp) {
        importance <- TRUE
        impmat <- matrix(0, p, n)
    } else impmat <- double(1)

    if (importance) {
        if (nPerm < 1) nPerm <- as.integer(1) else nPerm <- as.integer(nPerm)
        if (classRF) {
            impout <- matrix(0.0, p, nclass + 2)
            impSD <- matrix(0.0, p, nclass + 1)
        } else {
            impout <- matrix(0.0, p, 2)
            impSD <- double(p)
            names(impSD) <- x.col.names
        }
    } else {
        impout <- double(p)
        impSD <- double(1)
    }

    nsample <- if (addclass) 2 * n else n
    Stratify <- length(sampsize) > 1
    if ((!Stratify) && sampsize > nrow(x)) stop("sampsize too large")
    if (Stratify && (!classRF)) stop("sampsize should be of length one")
    if (classRF) {
        if (Stratify) {
            if (missing(strata)) strata <- y
            if (!is.factor(strata)) strata <- as.factor(strata)
            nsum <- sum(sampsize)
            if (length(sampsize) > nlevels(strata))
                stop("sampsize has too many elements.")
            if (any(sampsize <= 0) || nsum == 0)
                stop("Bad sampsize specification")
            ## If sampsize has names, match to class labels.
            if (!is.null(names(sampsize))) {
                sampsize <- sampsize[levels(strata)]
            }
            if (any(sampsize > table(strata)))
              stop("sampsize can not be larger than class frequency")
        } else {
            nsum <- sampsize
        }
        nrnodes <- 2 * trunc(nsum / nodesize) + 1
    } else {
        ## For regression trees, need to do this to get maximal trees.
        nrnodes <- 2 * trunc(sampsize/max(1, nodesize - 4)) + 1
    }
    if (!is.null(maxnodes)) {
        ## convert # of terminal nodes to total # of nodes
        maxnodes <- 2 * maxnodes - 1
        if (maxnodes > nrnodes) warning("maxnodes exceeds its max value.")
        nrnodes <- min(c(nrnodes, max(c(maxnodes, 1))))
    }
    ## Compiled code expects variables in rows and observations in columns.
    x <- t(x)
    storage.mode(x) <- "double"
    if (testdat) {
        xtest <- t(xtest)
        storage.mode(xtest) <- "double"
        if (is.null(ytest)) {
            ytest <- labelts <- 0
        } else {
            labelts <- TRUE
        }
    } else {
        xtest <- double(1)
        ytest <- double(1)
        ntest <- 1
        labelts <- FALSE
    }
    nt <- if (keep.forest) ntree else 1

    if (classRF) {
        cwt <- classwt
        threshold <- cutoff
        error.test <- if (labelts) double((nclass+1) * ntree) else double(1)
        rfout <- .C("classRF",
                    x = x,
                    xdim = as.integer(c(p, n)),
                    y = as.integer(y),
                    nclass = as.integer(nclass),
                    ncat = as.integer(ncat),
                    maxcat = as.integer(maxcat),
                    sampsize = as.integer(sampsize),
                    strata = if (Stratify) as.integer(strata) else integer(1),
                    Options = as.integer(c(addclass,
                    importance,
                    localImp,
                    proximity,
                    oob.prox,
                    do.trace,
                    keep.forest,
                    replace,
                    Stratify,
                    keep.inbag)),
                    ntree = as.integer(ntree),
                    mtry = as.integer(mtry),
                    ipi = as.integer(ipi),
                    classwt = as.double(cwt),
                    cutoff = as.double(threshold),
                    nodesize = as.integer(nodesize),
                    outcl = integer(nsample),
                    counttr = integer(nclass * nsample),
                    prox = prox,
                    impout = impout,
                    impSD = impSD,
                    impmat = impmat,
                    nrnodes = as.integer(nrnodes),
                    ndbigtree = integer(ntree),
                    nodestatus = integer(nt * nrnodes),
                    bestvar = integer(nt * nrnodes),
                    treemap = integer(nt * 2 * nrnodes),
                    nodepred = integer(nt * nrnodes),
                    xbestsplit = double(nt * nrnodes),
                    errtr = double((nclass+1) * ntree),
                    testdat = as.integer(testdat),
                    xts = as.double(xtest),
                    clts = as.integer(ytest),
                    nts = as.integer(ntest),
                    countts = double(nclass * ntest),
                    outclts = as.integer(numeric(ntest)),
                    labelts = as.integer(labelts),
                    proxts = proxts,
                    errts = error.test,
                    inbag = if (keep.inbag)
                    matrix(integer(n * ntree), n) else integer(n),
                    DUP=FALSE,
                    PACKAGE="randomForest")[-1]
        if (keep.forest) {
            ## deal with the random forest outputs
            max.nodes <- max(rfout$ndbigtree)
            treemap <- aperm(array(rfout$treemap, dim = c(2, nrnodes, ntree)),
                             c(2, 1, 3))[1:max.nodes, , , drop=FALSE]
        }
        if (!addclass) {
            ## Turn the predicted class into a factor like y.
            out.class <- factor(rfout$outcl, levels=1:nclass,
                                labels=levels(y))
            names(out.class) <- x.row.names
            con <- table(observed = y,
                         predicted = out.class)[levels(y), levels(y)]
            con <- cbind(con, class.error = 1 - diag(con)/rowSums(con))
        }
        out.votes <- t(matrix(rfout$counttr, nclass, nsample))[1:n, ]
        oob.times <- rowSums(out.votes)
        if (norm.votes)
            out.votes <- t(apply(out.votes, 1, function(x) x/sum(x)))
        dimnames(out.votes) <- list(x.row.names, levels(y))
        class(out.votes) <- c(class(out.votes), "votes")
        if (testdat) {
            out.class.ts <- factor(rfout$outclts, levels=1:nclass,
                                   labels=levels(y))
            names(out.class.ts) <- xts.row.names
            out.votes.ts <- t(matrix(rfout$countts, nclass, ntest))
            dimnames(out.votes.ts) <- list(xts.row.names, levels(y))
            if (norm.votes)
                out.votes.ts <- t(apply(out.votes.ts, 1,
                                        function(x) x/sum(x)))
            class(out.votes.ts) <- c(class(out.votes.ts), "votes")
            if (labelts) {
                testcon <- table(observed = ytest,
                                 predicted = out.class.ts)[levels(y), levels(y)]
                testcon <- cbind(testcon,
                                 class.error = 1 - diag(testcon)/rowSums(testcon))
            }
        }
        cl <- match.call()
        cl[[1]] <- as.name("randomForest")
        out <- list(call = cl,
                    type = if (addclass) "unsupervised" else "classification",
                    predicted = if (addclass) NULL else out.class,
                    err.rate = if (addclass) NULL else t(matrix(rfout$errtr,
                    nclass+1, ntree,
                    dimnames=list(c("OOB", levels(y)), NULL))),
                    confusion = if (addclass) NULL else con,
                    votes = out.votes,
                    oob.times = oob.times,
                    classes = levels(y),
                    importance = if (importance)
                    matrix(rfout$impout, p, nclass+2,
                           dimnames = list(x.col.names,
                           c(levels(y), "MeanDecreaseAccuracy",
                             "MeanDecreaseGini")))
                    else matrix(rfout$impout, ncol=1,
                                dimnames=list(x.col.names, "MeanDecreaseGini")),
                    importanceSD = if (importance)
                    matrix(rfout$impSD, p, nclass + 1,
                           dimnames = list(x.col.names,
                           c(levels(y), "MeanDecreaseAccuracy")))
                    else NULL,
                    localImportance = if (localImp)
                    matrix(rfout$impmat, p, n,
                           dimnames = list(x.col.names,x.row.names)) else NULL,
                    proximity = if (proximity) matrix(rfout$prox, n, n,
                    dimnames = list(x.row.names, x.row.names)) else NULL,
                    ntree = ntree,
                    mtry = mtry,
                    forest = if (!keep.forest) NULL else {
                        list(ndbigtree = rfout$ndbigtree,
                             nodestatus = matrix(rfout$nodestatus,
                             ncol = ntree)[1:max.nodes,, drop=FALSE],
                             bestvar = matrix(rfout$bestvar, ncol = ntree)[1:max.nodes,, drop=FALSE],
                             treemap = treemap,
                             nodepred = matrix(rfout$nodepred,
                             ncol = ntree)[1:max.nodes,, drop=FALSE],
                             xbestsplit = matrix(rfout$xbestsplit,
                             ncol = ntree)[1:max.nodes,, drop=FALSE],
                             pid = rfout$classwt, cutoff=cutoff, ncat=ncat,
                             maxcat = maxcat,
                             nrnodes = max.nodes, ntree = ntree,
                             nclass = nclass, xlevels=xlevels)
                    },
                    y = if (addclass) NULL else y,
                    test = if(!testdat) NULL else list(
                    predicted = out.class.ts,
                    err.rate = if (labelts) t(matrix(rfout$errts, nclass+1,
                    ntree,
                dimnames=list(c("Test", levels(y)), NULL))) else NULL,
                    confusion = if (labelts) testcon else NULL,
                    votes = out.votes.ts,
                    proximity = if(proximity) matrix(rfout$proxts, nrow=ntest,
                    dimnames = list(xts.row.names, c(xts.row.names,
                    x.row.names))) else NULL),
                    inbag = if (keep.inbag) matrix(rfout$inbag, nrow=nrow(rfout$inbag), 
										dimnames=list(x.row.names, NULL)) else NULL)
    } else {
		ymean <- mean(y)
		y <- y - ymean
		ytest <- ytest - ymean
        rfout <- .C("regRF",
                    x,
                    as.double(y),
                    as.integer(c(n, p)),
                    as.integer(sampsize),
                    as.integer(nodesize),
                    as.integer(nrnodes),
                    as.integer(ntree),
                    as.integer(mtry),
                    as.integer(c(importance, localImp, nPerm)),
                    as.integer(ncat),
                    as.integer(maxcat),
                    as.integer(do.trace),
                    as.integer(proximity),
                    as.integer(oob.prox),
                    as.integer(corr.bias),
                    ypred = double(n),
                    impout = impout,
                    impmat = impmat,
                    impSD = impSD,
                    prox = prox,
                    ndbigtree = integer(ntree),
                    nodestatus = matrix(integer(nrnodes * nt), ncol=nt),
                    leftDaughter = matrix(integer(nrnodes * nt), ncol=nt),
                    rightDaughter = matrix(integer(nrnodes * nt), ncol=nt),
                    nodepred = matrix(double(nrnodes * nt), ncol=nt),
                    bestvar = matrix(integer(nrnodes * nt), ncol=nt),
                    xbestsplit = matrix(double(nrnodes * nt), ncol=nt),
                    mse = double(ntree),
                    keep = as.integer(c(keep.forest, keep.inbag)),
                    replace = as.integer(replace),
                    testdat = as.integer(testdat),
                    xts = xtest,
                    ntest = as.integer(ntest),
                    yts = as.double(ytest),
                    labelts = as.integer(labelts),
                    ytestpred = double(ntest),
                    proxts = proxts,
                    msets = double(if (labelts) ntree else 1),
                    coef = double(2),
                    oob.times = integer(n),
                    inbag = if (keep.inbag)
                    matrix(integer(n * ntree), n) else integer(1),
                    DUP=FALSE,
                    PACKAGE="randomForest")[c(16:28, 36:41)]
        ## Format the forest component, if present.
        if (keep.forest) {
            max.nodes <- max(rfout$ndbigtree)
            rfout$nodestatus <-
                rfout$nodestatus[1:max.nodes, , drop=FALSE]
            rfout$bestvar <-
                rfout$bestvar[1:max.nodes, , drop=FALSE]
            rfout$nodepred <-
                rfout$nodepred[1:max.nodes, , drop=FALSE] + ymean
            rfout$xbestsplit <-
                rfout$xbestsplit[1:max.nodes, , drop=FALSE]
            rfout$leftDaughter <-
                rfout$leftDaughter[1:max.nodes, , drop=FALSE]
            rfout$rightDaughter <-
                rfout$rightDaughter[1:max.nodes, , drop=FALSE]
        }
        cl <- match.call()
        cl[[1]] <- as.name("randomForest")
        ## Make sure those obs. that have not been OOB get NA as prediction.
        ypred <- rfout$ypred
        if (any(rfout$oob.times < 1)) {
            ypred[rfout$oob.times == 0] <- NA
        }
        out <- list(call = cl,
                    type = "regression",
                    predicted = structure(ypred + ymean, names=x.row.names),
                    mse = rfout$mse,
                    rsq = 1 - rfout$mse / (var(y) * (n-1) / n),
                    oob.times = rfout$oob.times,
                    importance = if (importance) matrix(rfout$impout, p, 2,
                    dimnames=list(x.col.names,
                                  c("%IncMSE","IncNodePurity"))) else
                        matrix(rfout$impout, ncol=1,
                               dimnames=list(x.col.names, "IncNodePurity")),
                    importanceSD=if (importance) rfout$impSD else NULL,
                    localImportance = if (localImp)
                    matrix(rfout$impmat, p, n, dimnames=list(x.col.names,
                                               x.row.names)) else NULL,
                    proximity = if (proximity) matrix(rfout$prox, n, n,
                    dimnames = list(x.row.names, x.row.names)) else NULL,
                   ntree = ntree,
                    mtry = mtry,
                    forest = if (keep.forest)
                    c(rfout[c("ndbigtree", "nodestatus", "leftDaughter",
                              "rightDaughter", "nodepred", "bestvar",
                              "xbestsplit")],
                      list(ncat = ncat), list(nrnodes=max.nodes),
                      list(ntree=ntree), list(xlevels=xlevels)) else NULL,
                    coefs = if (corr.bias) rfout$coef else NULL,
                    y = y + ymean,
                    test = if(testdat) {
                        list(predicted = structure(rfout$ytestpred + ymean,
                             names=xts.row.names),
                             mse = if(labelts) rfout$msets else NULL,
                             rsq = if(labelts) 1 - rfout$msets /
                                        (var(ytest) * (n-1) / n) else NULL,
                             proximity = if (proximity)
                             matrix(rfout$proxts / ntree, nrow = ntest,
                                    dimnames = list(xts.row.names,
                                    c(xts.row.names,
                                    x.row.names))) else NULL)
                    } else NULL,
                    inbag = if (keep.inbag)
                    matrix(rfout$inbag, nrow(rfout$inbag),
                           dimnames=list(x.row.names, NULL)) else NULL)
    }
    class(out) <- "randomForest"
    return(out)
}
