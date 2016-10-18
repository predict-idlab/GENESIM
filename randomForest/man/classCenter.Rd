\name{classCenter}
\alias{classCenter}
\title{Prototypes of groups.}
\description{
  Prototypes are `representative' cases of a group of data points, given
  the similarity matrix among the points.  They are very similar to
  medoids.  The function is named `classCenter' to avoid conflict with
  the function \code{prototype} in the \code{methods} package.
}
\usage{
classCenter(x, label, prox, nNbr = min(table(label))-1) 
}
\arguments{
  \item{x}{a matrix or data frame}
  \item{label}{group labels of the rows in \code{x}}
  \item{prox}{the proximity (or similarity) matrix, assumed to be
    symmetric with 1 on the diagonal and in [0, 1] off the diagonal (the
  order of row/column must match that of \code{x})}
  \item{nNbr}{number of nearest neighbors used to find the prototypes.}
}
\value{
  A data frame containing one prototype in each row.
}
\details{
  This version only computes one prototype per class.  For each case in
  \code{x}, the \code{nNbr} nearest neighors are found.  Then, for each
  class, the case that has most neighbors of that class is identified.
  The prototype for that class is then the medoid of these neighbors
  (coordinate-wise medians for numerical variables and modes for
  categorical variables).

  This version only computes one prototype per class.  In the future
  more prototypes may be computed (by removing the `neighbors' used,
  then iterate).
}
%\references{}
\seealso{
  \code{\link{randomForest}}, \code{\link{MDSplot}}
}
\examples{
data(iris)
iris.rf <- randomForest(iris[,-5], iris[,5], prox=TRUE)
iris.p <- classCenter(iris[,-5], iris[,5], iris.rf$prox)
plot(iris[,3], iris[,4], pch=21, xlab=names(iris)[3], ylab=names(iris)[4],
     bg=c("red", "blue", "green")[as.numeric(factor(iris$Species))],
     main="Iris Data with Prototypes")
points(iris.p[,3], iris.p[,4], pch=21, cex=2, bg=c("red", "blue", "green"))
}
\author{Andy Liaw}
\keyword{classif}
