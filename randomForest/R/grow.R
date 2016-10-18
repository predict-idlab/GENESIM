grow <- function(x, ...) UseMethod("grow")

grow.default <- function(x, ...)
  stop("grow has not been implemented for this class of object")

grow.randomForest <- function(x, how.many, ...) {
  y <- update(x, ntree=how.many)
  combine(x, y)
}
