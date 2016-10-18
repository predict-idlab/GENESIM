importance <- function(x, ...)  UseMethod("importance")

importance.default <- function(x, ...)
    stop("No method implemented for this class of object")

importance.randomForest <- function(x, type=NULL, class=NULL, scale=TRUE,
                                    ...) {
    if (!inherits(x, "randomForest"))
        stop("x is not of class randomForest")
    classRF <- x$type != "regression"
    hasImp <- !is.null(dim(x$importance)) || ncol(x$importance) == 1
    hasType <- !is.null(type)
    if (hasType && type == 1 && !hasImp)
        stop("That measure has not been computed")
    allImp <- is.null(type) && hasImp
    if (hasType) {
        if (!(type %in% 1:2)) stop("Wrong type specified")
        if (type == 2 && !is.null(class))
            stop("No class-specific measure for that type")
    }
    
    imp <- x$importance
    if (hasType && type == 2) {
        if (hasImp) imp <- imp[, ncol(imp), drop=FALSE]
    } else {
        if (scale) {
            SD <- x$importanceSD
            imp[, -ncol(imp)] <-
                imp[, -ncol(imp), drop=FALSE] /
                    ifelse(SD < .Machine$double.eps, 1, SD)
        }
        if (!allImp) {
            if (is.null(class)) {
                ## The average decrease in accuracy measure:
                imp <- imp[, ncol(imp) - 1, drop=FALSE]
            } else {
                whichCol <- if (classRF) match(class, colnames(imp)) else 1
                if (is.na(whichCol)) stop(paste("Class", class, "not found."))
                imp <- imp[, whichCol, drop=FALSE]
            }
        }
    }
    imp
}
