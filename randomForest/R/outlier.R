outlier <- function(x, ...) UseMethod("outlier")

outlier.randomForest <- function(x, ...) {
    if (!inherits(x, "randomForest")) stop("x is not a randomForest object")
    if (x$type == "regression") stop("no outlier measure for regression")
    if (is.null(x$proximity)) stop("no proximity measures available")
    outlier.default(x$proximity, x$y)
}

outlier.default <- function(x, cls=NULL, ...) {
    if (nrow(x) != ncol(x)) stop ("x must be a square matrix")
    n <- nrow(x)
    if (is.null(cls)) cls <- rep(1, n)
    cls <- factor(cls)
    lvl <- levels(cls)
    cls.n <- table(cls)[lvl]
    id <- if (is.null(rownames(x))) 1:n else rownames(x)
    outlier <- structure(rep(NA, n), names=id)
    for (i in lvl) {
        iclass <- cls == i
        out <- rowSums(x[iclass, iclass]^2)
        out <- n / ifelse(out == 0, 1, out)
        out <- (out - median(out)) / mad(out)
        outlier[iclass] <- out
    }
    outlier
}
