margin <- function(x, ...) {
    UseMethod("margin")
}

margin.randomForest <- function(x, ...) {
    if (x$type == "regression") {
        stop("margin not defined for regression Random Forests")
    }
    if( is.null(x$votes) ) {
        stop("margin is only defined if votes are present")
    }
    margin(x$votes, x$y, ...)
}

margin.default <- function(x, observed, ...) {
    if ( !is.factor(observed) ) {
        stop(deparse(substitute(observed)), " is not a factor")
    }
    if (ncol(x) != nlevels(observed))
        stop("number of columns in x must equal the number of levels in observed")
    if (! all(colnames(x) %in% levels(observed)) ||
        ! all(levels(observed) %in% colnames(x)))
        stop("column names of x must match levels of observed")
    ## If the votes are not in fractions, normalize them to fractions.
    if ( any(x > 1) ) x <- sweep(x, 1, rowSums(x), "/")
    position <- match(as.character(observed), colnames(x))
    margin <- numeric(length(observed))
    for (i in seq_along(observed)) {
        margin[i] <- x[i, position[i]] - max(x[i, -position[i]])
    }
    names(margin) <- observed
    class(margin) <- "margin"
    margin
}

plot.margin <- function(x, sort=TRUE, ...) {
    if (sort) x <- sort(x)
    nF <- factor(names(x))
    nlevs <- length(levels(nF))
    if ( requireNamespace("RColorBrewer", quietly=TRUE) && nlevs < 12) {
        pal <- RColorBrewer::brewer.pal(nlevs,"Set1")
    } else {
        pal <- rainbow(nlevs)
    }
    plot.default(x, col=pal[as.numeric(nF)], pch=20, ... )
}

