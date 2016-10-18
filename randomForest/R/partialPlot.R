partialPlot <- function(x, ...) UseMethod("partialPlot")

partialPlot.default <- function(x, ...)
    stop("partial dependence plot not implemented for this class of objects.\n")

partialPlot.randomForest <-
    function (x, pred.data, x.var, which.class, w, plot=TRUE, add=FALSE,
              n.pt = min(length(unique(pred.data[, xname])), 51), rug = TRUE,
              xlab=deparse(substitute(x.var)), ylab="",
              main=paste("Partial Dependence on", deparse(substitute(x.var))),
              ...)
{
    classRF <- x$type != "regression"
    if (is.null(x$forest))
        stop("The randomForest object must contain the forest.\n")
    x.var <- substitute(x.var)
    xname <- if (is.character(x.var)) x.var else {
        if (is.name(x.var)) deparse(x.var) else {
            eval(x.var)
        }
    }
    xv <- pred.data[, xname]
    n <- nrow(pred.data)
    if (missing(w)) w <- rep(1, n)
    if (classRF) {
        if (missing(which.class)) {
            focus <- 1
        }
        else {
            focus <- charmatch(which.class, colnames(x$votes))
            if (is.na(focus))
                stop(which.class, "is not one of the class labels.")
        }
    }
    if (is.factor(xv) && !is.ordered(xv)) {
        x.pt <- levels(xv)
        y.pt <- numeric(length(x.pt))
        for (i in seq(along = x.pt)) {
            x.data <- pred.data
            x.data[, xname] <- factor(rep(x.pt[i], n), levels = x.pt)
            if (classRF) {
                pr <- predict(x, x.data, type = "prob")
                y.pt[i] <- weighted.mean(log(ifelse(pr[, focus] > 0,
                                                    pr[, focus], .Machine$double.eps)) -
                                         rowMeans(log(ifelse(pr > 0, pr, .Machine$double.eps))),
                                         w, na.rm=TRUE)
            } else y.pt[i] <- weighted.mean(predict(x, x.data), w, na.rm=TRUE)

        }
        if (add) {
            points(1:length(x.pt), y.pt, type="h", lwd=2, ...)
        } else {
            if (plot) barplot(y.pt, width=rep(1, length(y.pt)), col="blue",
                              xlab = xlab, ylab = ylab, main=main,
                              names.arg=x.pt, ...)
        }
    } else {
        if (is.ordered(xv)) xv <- as.numeric(xv)
        x.pt <- seq(min(xv), max(xv), length = n.pt)
        y.pt <- numeric(length(x.pt))
        for (i in seq(along = x.pt)) {
            x.data <- pred.data
            x.data[, xname] <- rep(x.pt[i], n)
            if (classRF) {
                pr <- predict(x, x.data, type = "prob")
                y.pt[i] <- weighted.mean(log(ifelse(pr[, focus] == 0,
                                                    .Machine$double.eps, pr[, focus]))
                                         - rowMeans(log(ifelse(pr == 0, .Machine$double.eps, pr))),
                                         w, na.rm=TRUE)
            } else {
                y.pt[i] <- weighted.mean(predict(x, x.data), w, na.rm=TRUE)
            }
        }
        if (add) {
            lines(x.pt, y.pt, ...)
        } else {
            if (plot) plot(x.pt, y.pt, type = "l", xlab=xlab, ylab=ylab,
                           main = main, ...)
        }
        if (rug && plot) {
            if (n.pt > 10) {
                rug(quantile(xv, seq(0.1, 0.9, by = 0.1)), side = 1)
            } else {
                rug(unique(xv, side = 1))
            }
        }
    }
    invisible(list(x = x.pt, y = y.pt))
}
