\name{outlier}
\alias{outlier}
\alias{outlier.randomForest}
\alias{outlier.default}
\title{Compute outlying measures}
\description{Compute outlying measures based on a proximity matrix.}
\usage{
\method{outlier}{default}(x, cls=NULL, ...)
\method{outlier}{randomForest}(x, ...)
}
\arguments{
  \item{x}{a proximity matrix (a square matrix with 1 on the diagonal
    and values between 0 and 1 in the off-diagonal positions); or an object of
    class \code{\link{randomForest}}, whose \code{type} is not
    \code{regression}.}
  \item{cls}{the classes the rows in the proximity matrix belong to.  If
    not given, all data are assumed to come from the same class.}
  \item{...}{arguments for other methods.}
}
\value{
  A numeric vector containing the outlying measures.  The outlying
  measure of a case is computed as n / sum(squared proximity), normalized by
  subtracting the median and divided by the MAD, within each class.
}
\seealso{
  \code{\link{randomForest}}
}
\examples{
set.seed(1)
iris.rf <- randomForest(iris[,-5], iris[,5], proximity=TRUE)
plot(outlier(iris.rf), type="h",
     col=c("red", "green", "blue")[as.numeric(iris$Species)])
}
\keyword{classif}
