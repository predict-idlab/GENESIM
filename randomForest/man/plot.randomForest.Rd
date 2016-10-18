\name{plot.randomForest}
\alias{plot.randomForest}
\title{Plot method for randomForest objects}
\description{
  Plot the error rates or MSE of a randomForest object
}
\usage{
\method{plot}{randomForest}(x, type="l", main=deparse(substitute(x)), ...)
}
\arguments{
  \item{x}{an object of class \code{randomForest}.}
  \item{type}{type of plot.}
  \item{main}{main title of the plot.}
  \item{...}{other graphical parameters.}
}
\value{
  Invisibly, the error rates or MSE of the \code{randomForest} object.
  If the object has a non-null \code{test} component, then the returned
  object is a matrix where the first column is the out-of-bag estimate
  of error, and the second column is for the test set.
}
\note{
  This function does not work for \code{randomForest} objects that have
  \code{type=unsupervised}.
  
  If the \code{x} has a non-null \code{test} component, then the test
  set errors are also plotted.
}
\seealso{
  \code{\link{randomForest}}
}
\examples{
data(mtcars)
plot(randomForest(mpg ~ ., mtcars, keep.forest=FALSE, ntree=100), log="y")
}
\author{Andy Liaw}
\keyword{classif}
\keyword{regression}
\keyword{tree}
