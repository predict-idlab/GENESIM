rfImpute <- function(x, ...)
    UseMethod("rfImpute")

rfImpute.formula <- function(x, data, ..., subset) {
    if (!inherits(x, "formula"))
        stop("method is only for formula objects")
    call <- match.call()
    m <- match.call(expand.dots = FALSE)
    names(m)[2] <- "formula"
    if (is.matrix(eval(m$data, parent.frame())))
        m$data <- as.data.frame(data)
    m$... <- NULL
    m$na.action <- as.name("na.pass")
    m[[1]] <- as.name("model.frame")
    m <- eval(m, parent.frame())
    Terms <- attr(m, "terms")
    attr(Terms, "intercept") <- 0
    y <- model.response(m)
    if (!is.null(y)) m <- m[,-1]
    for (i in seq(along=ncol(m))) {
        if(is.ordered(m[[i]])) m[[i]] <- as.numeric(m[[i]])
    }
    ret <- rfImpute.default(m, y, ...)
    names(ret)[1] <- deparse(as.list(x)[[2]])
    ret
}

rfImpute.default <- function(x, y, iter=5, ntree=300, ...) {
    if (any(is.na(y))) stop("Can't have NAs in", deparse(substitute(y)))
    if (!any(is.na(x))) stop("No NAs found in ", deparse(substitute(x)))
    xf <- na.roughfix(x)
    hasNA <- which(apply(x, 2, function(x) any(is.na(x))))
    if (is.data.frame(x)) {
        isfac <- sapply(x, is.factor)
    } else {
        isfac <- rep(FALSE, ncol(x))
    }
    
    for (i in 1:iter) {
        prox <- randomForest(xf, y, ntree=ntree, ..., do.trace=ntree,
                             proximity=TRUE)$proximity
        for (j in hasNA) {
            miss <- which(is.na(x[, j]))
            if (isfac[j]) {
                lvl <- levels(x[[j]])
                catprox <- apply(prox[-miss, miss, drop=FALSE], 2,
                                 function(v) lvl[which.max(tapply(v, x[[j]][-miss], mean))])
                xf[miss, j] <- catprox
            } else {
                sumprox <- colSums(prox[-miss, miss, drop=FALSE])
                xf[miss, j] <- (prox[miss, -miss, drop=FALSE] %*% xf[,j][-miss]) / (1e-8 + sumprox)
            }
            NULL
        }
    }
    xf <- cbind(y, xf)
    names(xf)[1] <- deparse(substitute(y))
    xf
}
