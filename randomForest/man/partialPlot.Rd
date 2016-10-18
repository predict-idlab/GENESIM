\name{partialPlot}
\alias{partialPlot}
\alias{partialPlot.default}
\alias{partialPlot.randomForest}

\title{Partial dependence plot}

\description{
Partial dependence plot gives a graphical depiction of the marginal
effect of a variable on the class probability (classification) or
response (regression).
}

\usage{
\method{partialPlot}{randomForest}(x, pred.data, x.var, which.class,
      w, plot = TRUE, add = FALSE,
      n.pt = min(length(unique(pred.data[, xname])), 51),
      rug = TRUE, xlab=deparse(substitute(x.var)), ylab="",
      main=paste("Partial Dependence on", deparse(substitute(x.var))),
      ...)
}

\arguments{
  \item{x}{an object of class \code{randomForest}, which contains a
    \code{forest} component.}
  \item{pred.data}{a data frame used for contructing the plot, usually
    the training data used to contruct the random forest.}
  \item{x.var}{name of the variable for which partial
    dependence is to be examined.}
  \item{which.class}{For classification data, the class to focus on
    (default the first class).}
  \item{w}{weights to be used in averaging; if not supplied, mean is not
  weighted}
  \item{plot}{whether the plot should be shown on the graphic device.}
  \item{add}{whether to add to existing plot (\code{TRUE}).}
  \item{n.pt}{if \code{x.var} is continuous, the number of points on the
    grid for evaluating partial dependence.}
  \item{rug}{whether to draw hash marks at the bottom of the plot
    indicating the deciles of \code{x.var}.}
  \item{xlab}{label for the x-axis.}
  \item{ylab}{label for the y-axis.}
  \item{main}{main title for the plot.}
  \item{...}{other graphical parameters to be passed on to \code{plot}
    or \code{lines}.}
}

\value{
A list with two components: \code{x} and \code{y}, which are the values
used in the plot.
}

\details{
  The function being plotted is defined as:
  \deqn{
    \tilde{f}(x) = \frac{1}{n} \sum_{i=1}^n f(x, x_{iC}),
  }
  where \eqn{x} is the variable for which partial dependence is sought,
  and \eqn{x_{iC}} is the other variables in the data.  The summand is
  the predicted regression function for regression, and logits
  (i.e., log of fraction of votes) for \code{which.class} for
  classification:
\deqn{ f(x) = \log p_k(x) - \frac{1}{K} \sum_{j=1}^K \log p_j(x),}
where \eqn{K} is the number of classes, \eqn{k} is \code{which.class},
and \eqn{p_j} is the proportion of votes for class \eqn{j}.
}
\note{
  The \code{randomForest} object must contain the \code{forest}
  component; i.e., created with \code{randomForest(...,
    keep.forest=TRUE)}.

  This function runs quite slow for large data sets.
}
\references{
Friedman, J. (2001). Greedy function approximation: the gradient
boosting machine, \emph{Ann. of Stat.}}

\seealso{\code{\link{randomForest}}}

\author{Andy Liaw \email{andy\_liaw@merck.com}}

\examples{
data(iris)
set.seed(543)
iris.rf <- randomForest(Species~., iris)
partialPlot(iris.rf, iris, Petal.Width, "versicolor")

## Looping over variables ranked by importance:
data(airquality)
airquality <- na.omit(airquality)
set.seed(131)
ozone.rf <- randomForest(Ozone ~ ., airquality, importance=TRUE)
imp <- importance(ozone.rf)
impvar <- rownames(imp)[order(imp[, 1], decreasing=TRUE)]
op <- par(mfrow=c(2, 3))
for (i in seq_along(impvar)) {
    partialPlot(ozone.rf, airquality, impvar[i], xlab=impvar[i],
                main=paste("Partial Dependence on", impvar[i]),
                ylim=c(30, 70))
}
par(op)
}
\keyword{classif}
\keyword{regression}
\keyword{tree}

