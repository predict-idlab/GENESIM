\name{getTree}
\alias{getTree}
\title{Extract a single tree from a forest.}
\description{
  This function extract the structure of a tree from a
  \code{randomForest} object.
}
\usage{
getTree(rfobj, k=1, labelVar=FALSE)
}
\arguments{
  \item{rfobj}{a \code{\link{randomForest}} object.}
  \item{k}{which tree to extract?}
  \item{labelVar}{Should better labels be used for splitting variables
    and predicted class?}
}
\value{
  A matrix (or data frame, if \code{labelVar=TRUE}) with six columns and
  number of rows equal to total number of nodes in the tree.  The six
  columns are:
  \item{left daughter}{the row where the left daughter node is; 0 if the
  node is terminal}
  \item{right daughter}{the row where the right daughter node is; 0 if
    the node is terminal}
  \item{split var}{which variable was used to split the node; 0 if the
    node is terminal}
  \item{split point}{where the best split is; see Details for
    categorical predictor}
  \item{status}{is the node terminal (-1) or not (1)}
  \item{prediction}{the prediction for the node; 0 if the node is not
    terminal}
}
\details{
  For numerical predictors, data with values of the variable less than
  or equal to the splitting point go to the left daughter node.

  For categorical predictors, the splitting point is represented by an
  integer, whose binary expansion gives the identities of the categories
  that goes to left or right.  For example, if a predictor has four
  categories, and the split point is 13.  The binary expansion of 13 is
  (1, 0, 1, 1) (because \eqn{13 = 1*2^0 + 0*2^1 + 1*2^2 + 1*2^3}), so cases with
  categories 1, 3, or 4 in this predictor get sent to the left, and the rest
  to the right.
}
%\references{
%}
\seealso{
  \code{\link{randomForest}}
}
\examples{
data(iris)
## Look at the third trees in the forest.
getTree(randomForest(iris[,-5], iris[,5], ntree=10), 3, labelVar=TRUE)
}
\author{Andy Liaw \email{andy\_liaw@merck.com}}
\keyword{tree}
