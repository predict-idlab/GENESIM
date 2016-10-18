\name{predict.randomForest}
\alias{predict.randomForest}
\title{predict method for random forest objects}
\description{
  Prediction of test data using random forest.
}
\usage{
\method{predict}{randomForest}(object, newdata, type="response",
  norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE,
  cutoff, ...)
}
\arguments{
  \item{object}{an object of class \code{randomForest}, as that
    created by the function \code{randomForest}.}
  \item{newdata}{a data frame or matrix containing new data.  (Note: If
    not given, the out-of-bag prediction in \code{object} is returned.}
  \item{type}{one of \code{response}, \code{prob}. or \code{votes},
  indicating the type of output: predicted values, matrix of class
  probabilities, or matrix of vote counts.  \code{class} is allowed, but
  automatically converted to "response", for backward compatibility.}
  \item{norm.votes}{Should the vote counts be normalized (i.e.,
    expressed as fractions)?  Ignored if \code{object$type} is
    \code{regression}.}
  \item{predict.all}{Should the predictions of all trees be kept?}
  \item{proximity}{Should proximity measures be computed?  An error is
    issued if \code{object$type} is \code{regression}.}
  \item{nodes}{Should the terminal node indicators (an n by ntree
    matrix) be return?  If so, it is in the ``nodes'' attribute of the
    returned object.}
  \item{cutoff}{(Classification only)  A vector of length equal to
    number of classes.  The `winning' class for an observation is the
    one with the maximum ratio of proportion of votes to cutoff.
    Default is taken from the \code{forest$cutoff} component of
    \code{object} (i.e., the setting used when running
    \code{\link{randomForest}}).}
  \item{...}{not used currently.}
}

\value{
  If \code{object$type} is \code{regression}, a vector of predicted
  values is returned.  If \code{predict.all=TRUE}, then the returned
  object is a list of two components: \code{aggregate}, which is the
  vector of predicted values by the forest, and \code{individual}, which
  is a matrix where each column contains prediction by a tree in the
  forest.

  If \code{object$type} is \code{classification}, the object returned
  depends on the argument \code{type}:
  \item{response}{predicted classes (the classes with majority vote).}
  \item{prob}{matrix of class probabilities (one column for each class
  and one row for each input).}
  \item{vote}{matrix of vote counts (one column for each class
  and one row for each new input); either in raw counts or in fractions
  (if \code{norm.votes=TRUE}).}

If \code{predict.all=TRUE}, then the \code{individual} component of the
returned object is a character matrix where each column contains the
predicted class by a tree in the forest.

If \code{proximity=TRUE}, the returned object is a list with two
components: \code{pred} is the prediction (as described above) and
\code{proximity} is the proximitry matrix.  An error is issued if
\code{object$type} is \code{regression}.

If \code{nodes=TRUE}, the returned object has a ``nodes'' attribute,
which is an n by ntree matrix, each column containing the node number
that the cases fall in for that tree.

NOTE: If the \code{object} inherits from \code{randomForest.formula},
then any data with \code{NA} are silently omitted from the prediction.
The returned value will contain \code{NA} correspondingly in the
aggregated and individual tree predictions (if requested), but not in
the proximity or node matrices.

NOTE2: Any ties are broken at random, so if this is undesirable, avoid it by
using odd number \code{ntree} in \code{randomForest()}.
}
\references{
  Breiman, L. (2001), \emph{Random Forests}, Machine Learning 45(1),
  5-32.
}
\author{ Andy Liaw \email{andy\_liaw@merck.com} and Matthew Wiener
  \email{matthew\_wiener@merck.com}, based on original Fortran code by
  Leo Breiman and Adele Cutler.}

\seealso{\code{\link{randomForest}}}

\examples{
data(iris)
set.seed(111)
ind <- sample(2, nrow(iris), replace = TRUE, prob=c(0.8, 0.2))
iris.rf <- randomForest(Species ~ ., data=iris[ind == 1,])
iris.pred <- predict(iris.rf, iris[ind == 2,])
table(observed = iris[ind==2, "Species"], predicted = iris.pred)
## Get prediction for all trees.
predict(iris.rf, iris[ind == 2,], predict.all=TRUE)
## Proximities.
predict(iris.rf, iris[ind == 2,], proximity=TRUE)
## Nodes matrix.
str(attr(predict(iris.rf, iris[ind == 2,], nodes=TRUE), "nodes"))
}
\keyword{classif}% at least one, from doc/KEYWORDS
\keyword{regression}
