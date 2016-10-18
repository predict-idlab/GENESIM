"randomForest.formula" <-
    function(formula, data = NULL, ..., subset, na.action = na.fail) {
### formula interface for randomForest.
### code gratefully stolen from svm.formula (package e1071).
###
    if (!inherits(formula, "formula"))
        stop("method is only for formula objects")
    m <- match.call(expand.dots = FALSE)
    ## Catch xtest and ytest in arguments.
    if (any(c("xtest", "ytest") %in% names(m)))
        stop("xtest/ytest not supported through the formula interface")
    names(m)[2] <- "formula"
    if (is.matrix(eval(m$data, parent.frame())))
        m$data <- as.data.frame(data)
    m$... <- NULL
    m$na.action <- na.action
    m[[1]] <- as.name("model.frame")
    m <- eval(m, parent.frame())
	#rn <- 1:nrow(m)
	
    y <- model.response(m)
    Terms <- attr(m, "terms")
    attr(Terms, "intercept") <- 0
	attr(y, "na.action") <- attr(m, "na.action")
	## Drop any "negative" terms in the formula.
    ## test with:
    ## randomForest(Fertility~.-Catholic+I(Catholic<50),data=swiss,mtry=2)
    m <- model.frame(terms(reformulate(attributes(Terms)$term.labels)),
                     data.frame(m))
    ## if (!is.null(y)) m <- m[, -1, drop=FALSE]
    for (i in seq(along=m)) {
        if (is.ordered(m[[i]])) m[[i]] <- as.numeric(m[[i]])
    }
    ret <- randomForest.default(m, y, ...)
    cl <- match.call()
    cl[[1]] <- as.name("randomForest")
    ret$call <- cl
    ret$terms <- Terms
    if (!is.null(attr(y, "na.action"))) {
        attr(ret$predicted, "na.action") <- ret$na.action <- attr(y, "na.action")
	}
    class(ret) <- c("randomForest.formula", "randomForest")
    return(ret)
}
