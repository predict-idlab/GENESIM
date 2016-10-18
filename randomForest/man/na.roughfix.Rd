\name{na.roughfix}
\alias{na.roughfix}
\alias{na.roughfix.default}
\alias{na.roughfix.data.frame}
\title{Rough Imputation of Missing Values}
\description{
  Impute Missing Values by median/mode.
}
\usage{
na.roughfix(object, ...)
}
\arguments{
  \item{object}{a data frame or numeric matrix.}
  \item{\dots}{further arguments special methods could require.}
}
\value{
  A completed data matrix or data frame.  For numeric variables,
  \code{NA}s are replaced with column medians.  For factor variables,
  \code{NA}s are replaced with the most frequent levels (breaking ties
  at random).  If \code{object} contains no \code{NA}s, it is returned
  unaltered. 
}
\note{
  This is used as a starting point for imputing missing values by random
  forest.
}
%\references{
%}
\seealso{
\code{\link{rfImpute}}, \code{\link{randomForest}}.
}
\examples{
data(iris)
iris.na <- iris
set.seed(111)
## artificially drop some data values.
for (i in 1:4) iris.na[sample(150, sample(20)), i] <- NA
iris.roughfix <- na.roughfix(iris.na)
iris.narf <- randomForest(Species ~ ., iris.na, na.action=na.roughfix)
print(iris.narf)
}
\author{Andy Liaw}
\keyword{NA}
