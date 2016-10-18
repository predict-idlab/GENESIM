\name{rfcv}
\alias{rfcv}
\title{Random Forest Cross-Valdidation for feature selection}
\description{
This function shows the cross-validated prediction performance of
models with sequentially reduced number of predictors (ranked by
variable importance) via a nested cross-validation procedure.
}
\usage{
rfcv(trainx, trainy, cv.fold=5, scale="log", step=0.5,
     mtry=function(p) max(1, floor(sqrt(p))), recursive=FALSE, ...)
}
\arguments{
  \item{trainx}{matrix or data frame containing columns of predictor
  variables}
  \item{trainy}{vector of response, must have length equal to the number
  of rows in \code{trainx}}
  \item{cv.fold}{number of folds in the cross-validation}
  \item{scale}{if \code{"log"}, reduce a fixed proportion (\code{step})
  of variables at each step, otherwise reduce \code{step} variables at a
  time}
  \item{step}{if \code{log=TRUE}, the fraction of variables to remove at
	each step, else remove this many variables at a time}
  \item{mtry}{a function of number of remaining predictor variables to
	use as the \code{mtry} parameter in the \code{randomForest} call}
  \item{recursive}{whether variable importance is (re-)assessed at each
  step of variable reduction}
  \item{...}{other arguments passed on to \code{randomForest}}
}
\value{
  A list with the following components:

    list(n.var=n.var, error.cv=error.cv, predicted=cv.pred)
	\item{n.var}{vector of number of variables used at each step}
	\item{error.cv}{corresponding vector of error rates or MSEs at each
	  step}
	\item{predicted}{list of \code{n.var} components, each containing
	  the predicted values from the cross-validation}
}
%\details{
%}
\references{
Svetnik, V., Liaw, A., Tong, C. and Wang, T., ``Application of Breiman's
Random Forest to Modeling Structure-Activity Relationships of
Pharmaceutical Molecules'', MCS 2004, Roli, F. and Windeatt, T. (Eds.)
pp. 334-343.
}
\seealso{
  \code{\link{randomForest}}, \code{\link{importance}}
}
\examples{
set.seed(647)
myiris <- cbind(iris[1:4], matrix(runif(96 * nrow(iris)), nrow(iris), 96))
result <- rfcv(myiris, iris$Species, cv.fold=3)
with(result, plot(n.var, error.cv, log="x", type="o", lwd=2))

## The following can take a while to run, so if you really want to try
## it, copy and paste the code into R.

\dontrun{
result <- replicate(5, rfcv(myiris, iris$Species), simplify=FALSE)
error.cv <- sapply(result, "[[", "error.cv")
matplot(result[[1]]$n.var, cbind(rowMeans(error.cv), error.cv), type="l",
        lwd=c(2, rep(1, ncol(error.cv))), col=1, lty=1, log="x",
        xlab="Number of variables", ylab="CV Error")
}
}
\author{Andy Liaw}
\keyword{classif}
\keyword{regression}

