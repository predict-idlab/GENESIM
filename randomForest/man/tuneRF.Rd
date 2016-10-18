\name{tuneRF}
\alias{tuneRF}
\title{Tune randomForest for the optimal mtry parameter}
\description{
  Starting with the default value of mtry, search for the optimal value
  (with respect to Out-of-Bag error estimate) of mtry for randomForest.
}
\usage{
tuneRF(x, y, mtryStart, ntreeTry=50, stepFactor=2, improve=0.05,
       trace=TRUE, plot=TRUE, doBest=FALSE, ...)
}
\arguments{
  \item{x}{matrix or data frame of predictor variables}
  \item{y}{response vector (factor for classification, numeric for
    regression)}
  \item{mtryStart}{starting value of mtry; default is the same as in
    \code{\link{randomForest}}}
  \item{ntreeTry}{number of trees used at the tuning step}
  \item{stepFactor}{at each iteration, mtry is inflated (or deflated) by
    this value}
  \item{improve}{the (relative) improvement in OOB error must be by this
    much for the search to continue}
  \item{trace}{whether to print the progress of the search}
  \item{plot}{whether to plot the OOB error as function of mtry}
  \item{doBest}{whether to run a forest using the optimal mtry found}
  \item{...}{options to be given to \code{\link{randomForest}}}
}
\value{
  If \code{doBest=FALSE} (default), it returns a matrix whose first
  column contains the mtry values searched, and the second column the
  corresponding OOB error.

  If \code{doBest=TRUE}, it returns the \code{\link{randomForest}}
  object produced with the optimal \code{mtry}.
}
%\details{
%}
%\references{
%}
\seealso{
\code{\link{randomForest}}
}
\examples{
data(fgl, package="MASS")
fgl.res <- tuneRF(fgl[,-10], fgl[,10], stepFactor=1.5)
}
%\author{}
\keyword{classif}
\keyword{tree}
