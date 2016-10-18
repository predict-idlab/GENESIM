"predict.randomForest" <-
    function (object, newdata, type = "response", norm.votes = TRUE,
              predict.all=FALSE, proximity = FALSE, nodes=FALSE, cutoff, ...)
{
    if (!inherits(object, "randomForest"))
        stop("object not of class randomForest")
    if (is.null(object$forest)) stop("No forest component in the object")
    out.type <- charmatch(tolower(type),
                          c("response", "prob", "vote", "class"))
    if (is.na(out.type))
        stop("type must be one of 'response', 'prob', 'vote'")
    if (out.type == 4) out.type <- 1
    if (out.type != 1 && object$type == "regression")
        stop("'prob' or 'vote' not meaningful for regression")
    if (out.type == 2)
        norm.votes <- TRUE
    if (missing(newdata)) {
		p <- if (! is.null(object$na.action)) {
			napredict(object$na.action, object$predicted)
		} else {
			object$predicted
		}
        if (object$type == "regression") return(p)
        if (proximity & is.null(object$proximity))
            warning("cannot return proximity without new data if random forest object does not already have proximity")
        if (out.type == 1) {
            if (proximity) {
                return(list(pred = p,
                            proximity = object$proximity))
            } else return(p)
        }
		v <- object$votes
		if (!is.null(object$na.action)) v <- napredict(object$na.action, v)
        if (norm.votes) {
            t1 <- t(apply(v, 1, function(x) { x/sum(x) }))
            class(t1) <- c(class(t1), "votes")
            if (proximity) return(list(pred = t1, proximity = object$proximity))
            else return(t1)
        } else {
            if (proximity) return(list(pred = v, proximity = object$proximity))
            else return(v)
        }
    }
    if (missing(cutoff)) {
        cutoff <- object$forest$cutoff
    } else {
        if (sum(cutoff) > 1 || sum(cutoff) < 0 || !all(cutoff > 0) ||
            length(cutoff) != length(object$classes)) {
            stop("Incorrect cutoff specified.")
        }
        if (!is.null(names(cutoff))) {
            if (!all(names(cutoff) %in% object$classes)) {
                stop("Wrong name(s) for cutoff")
            }
            cutoff <- cutoff[object$classes]
        }
    }

    if (object$type == "unsupervised")
        stop("Can't predict unsupervised forest.")

    if (inherits(object, "randomForest.formula")) {
        newdata <- as.data.frame(newdata)
        rn <- row.names(newdata)
        Terms <- delete.response(object$terms)
        x <- model.frame(Terms, newdata, na.action = na.omit)
        keep <- match(row.names(x), rn)
    } else {
        if (is.null(dim(newdata)))
            dim(newdata) <- c(1, length(newdata))
        x <- newdata
        if (nrow(x) == 0)
            stop("newdata has 0 rows")
        if (any(is.na(x)))
            stop("missing values in newdata")
        keep <- 1:nrow(x)
        rn <- rownames(x)
        if (is.null(rn)) rn <- keep
    }
    vname <- if (is.null(dim(object$importance))) {
        names(object$importance)
    } else {
        rownames(object$importance)
    }
    if (is.null(colnames(x))) {
        if (ncol(x) != length(vname)) {
            stop("number of variables in newdata does not match that in the training data")
        }
    } else {
        if (any(! vname %in% colnames(x)))
            stop("variables in the training data missing in newdata")
        x <- x[, vname, drop=FALSE]
    }
    if (is.data.frame(x)) {
		isFactor <- function(x) is.factor(x) & ! is.ordered(x)
        xfactor <- which(sapply(x, isFactor))
        if (length(xfactor) > 0 && "xlevels" %in% names(object$forest)) {
            for (i in xfactor) {
                if (any(! levels(x[[i]]) %in% object$forest$xlevels[[i]]))
                    stop("New factor levels not present in the training data")
                x[[i]] <-
                    factor(x[[i]],
                           levels=levels(x[[i]])[match(levels(x[[i]]), object$forest$xlevels[[i]])])
            }
        }
        cat.new <- sapply(x, function(x) if (is.factor(x) && !is.ordered(x))
                          length(levels(x)) else 1)
        if (!all(object$forest$ncat == cat.new))
            stop("Type of predictors in new data do not match that of the training data.")
    }
    mdim <- ncol(x)
    ntest <- nrow(x)
    ntree <- object$forest$ntree
    maxcat <- max(object$forest$ncat)
    nclass <- object$forest$nclass
    nrnodes <- object$forest$nrnodes
    ## get rid of warning:
    op <- options(warn=-1)
    on.exit(options(op))
    x <- t(data.matrix(x))

    if (predict.all) {
        treepred <- if (object$type == "regression") {
            matrix(double(ntest * ntree), ncol=ntree)
        } else {
            matrix(integer(ntest * ntree), ncol=ntree)
        }
    } else {
        treepred <- numeric(ntest)
    }
    proxmatrix <- if (proximity) matrix(0, ntest, ntest) else numeric(1)
    nodexts <- if (nodes) integer(ntest * ntree) else integer(ntest)

    if (object$type == "regression") {
            if (!is.null(object$forest$treemap)) {
                object$forest$leftDaughter <-
                    object$forest$treemap[,1,, drop=FALSE]
                object$forest$rightDaughter <-
                    object$forest$treemap[,2,, drop=FALSE]
                object$forest$treemap <- NULL
            }

            keepIndex <- "ypred"
            if (predict.all) keepIndex <- c(keepIndex, "treepred")
            if (proximity) keepIndex <- c(keepIndex, "proximity")
            if (nodes) keepIndex <- c(keepIndex, "nodexts")
            ## Ensure storage mode is what is expected in C.
            if (! is.integer(object$forest$leftDaughter))
                storage.mode(object$forest$leftDaughter) <- "integer"
            if (! is.integer(object$forest$rightDaughter))
                storage.mode(object$forest$rightDaughter) <- "integer"
            if (! is.integer(object$forest$nodestatus))
                storage.mode(object$forest$nodestatus) <- "integer"
            if (! is.double(object$forest$xbestsplit))
                storage.mode(object$forest$xbestsplit) <- "double"
            if (! is.double(object$forest$nodepred))
                storage.mode(object$forest$nodepred) <- "double"
            if (! is.integer(object$forest$bestvar))
                storage.mode(object$forest$bestvar) <- "integer"
            if (! is.integer(object$forest$ndbigtree))
                storage.mode(object$forest$ndbigtree) <- "integer"
            if (! is.integer(object$forest$ncat))
                storage.mode(object$forest$ncat) <- "integer"

            ans <- .C("regForest",
                  as.double(x),
                  ypred = double(ntest),
                  as.integer(mdim),
                  as.integer(ntest),
                  as.integer(ntree),
                  object$forest$leftDaughter,
                  object$forest$rightDaughter,
                  object$forest$nodestatus,
                  nrnodes,
                  object$forest$xbestsplit,
                  object$forest$nodepred,
                  object$forest$bestvar,
                  object$forest$ndbigtree,
                  object$forest$ncat,
                  as.integer(maxcat),
                  as.integer(predict.all),
                  treepred = as.double(treepred),
                  as.integer(proximity),
                  proximity = as.double(proxmatrix),
                  nodes = as.integer(nodes),
                  nodexts = as.integer(nodexts),
                  DUP=FALSE,
                  PACKAGE = "randomForest")[keepIndex]
            ## Apply bias correction if needed.
            yhat <- rep(NA, length(rn))
            names(yhat) <- rn
            if (!is.null(object$coefs)) {
                yhat[keep] <- object$coefs[1] + object$coefs[2] * ans$ypred
            } else {
                yhat[keep] <- ans$ypred
            }
            if (predict.all) {
                treepred <- matrix(NA, length(rn), ntree,
                                   dimnames=list(rn, NULL))
                treepred[keep,] <- ans$treepred
            }
            if (!proximity) {
                res <- if (predict.all)
                    list(aggregate=yhat, individual=treepred) else yhat
            } else {
                res <- list(predicted = yhat,
                            proximity = structure(ans$proximity,
                            dim=c(ntest, ntest), dimnames=list(rn, rn)))
            }
            if (nodes) {
                attr(res, "nodes") <- matrix(ans$nodexts, ntest, ntree,
                                             dimnames=list(rn[keep], 1:ntree))
            }
        } else {
        countts <- matrix(0, ntest, nclass)
        t1 <- .C("classForest",
                 mdim = as.integer(mdim),
                 ntest = as.integer(ntest),
                 nclass = as.integer(object$forest$nclass),
                 maxcat = as.integer(maxcat),
                 nrnodes = as.integer(nrnodes),
                 jbt = as.integer(ntree),
                 xts = as.double(x),
                 xbestsplit = as.double(object$forest$xbestsplit),
                 pid = object$forest$pid,
                 cutoff = as.double(cutoff),
                 countts = as.double(countts),
                 treemap = as.integer(aperm(object$forest$treemap,
                                 c(2, 1, 3))),
                 nodestatus = as.integer(object$forest$nodestatus),
                 cat = as.integer(object$forest$ncat),
                 nodepred = as.integer(object$forest$nodepred),
                 treepred = as.integer(treepred),
                 jet = as.integer(numeric(ntest)),
                 bestvar = as.integer(object$forest$bestvar),
                 nodexts = as.integer(nodexts),
                 ndbigtree = as.integer(object$forest$ndbigtree),
                 predict.all = as.integer(predict.all),
                 prox = as.integer(proximity),
                 proxmatrix = as.double(proxmatrix),
                 nodes = as.integer(nodes),
                 DUP=FALSE,
                 PACKAGE = "randomForest")
        if (out.type > 1) {
            out.class.votes <- t(matrix(t1$countts, nrow = nclass, ncol = ntest))
            if (norm.votes)
                out.class.votes <-
                    sweep(out.class.votes, 1, rowSums(out.class.votes), "/")
            z <- matrix(NA, length(rn), nclass,
                        dimnames=list(rn, object$classes))
            z[keep, ] <- out.class.votes
             class(z) <- c(class(z), "votes")
            res <- z
        } else {
            out.class <- factor(rep(NA, length(rn)),
                                levels=1:length(object$classes),
                                labels=object$classes)
            out.class[keep] <- object$classes[t1$jet]
            names(out.class)[keep] <- rn[keep]
            res <- out.class
        }
        if (predict.all) {
            treepred <- matrix(object$classes[t1$treepred],
                               nrow=length(keep), dimnames=list(rn[keep], NULL))
            res <- list(aggregate=res, individual=treepred)
        }
        if (proximity)
            res <- list(predicted = res, proximity = structure(t1$proxmatrix,
                                         dim = c(ntest, ntest),
                                         dimnames = list(rn[keep], rn[keep])))
        if (nodes) attr(res, "nodes") <- matrix(t1$nodexts, ntest, ntree,
                                                dimnames=list(rn[keep], 1:ntree))
    }
    res
}
