\name{measureRule}
\alias{measureRule}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal
}
\description{
internal
}
\usage{
measureRule(ruleExec, X, target, pred = NULL, regMethod = "mean")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleExec}{
%%     ~~Describe \code{ruleExec} here~~
}
  \item{X}{
%%     ~~Describe \code{X} here~~
}
  \item{target}{
%%     ~~Describe \code{target} here~~
}
  \item{pred}{
%%     ~~Describe \code{pred} here~~
}
  \item{regMethod}{
%%     ~~Describe \code{pred} here~~
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
% \value{
% %%  ~Describe the value returned
% %%  If it is a LIST, use
% %%  \item{comp1 }{Description of 'comp1'}
% %%  \item{comp2 }{Description of 'comp2'}
% %% ...
% }
% \references{
% %% ~put references to the literature/web site here ~
% }
% \author{
% %%  ~~who you are~~
% }
% \note{
% %%  ~~further notes~~
% }

% %% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (ruleExec, X, target, pred = NULL) 
{
    len <- length(unlist(strsplit(ruleExec, split = " & ")))
    origRule <- ruleExec
    ruleExec <- paste("which(", ruleExec, ")")
    ixMatch <- eval(parse(text = ruleExec))
    if (length(ixMatch) == 0) {
        v <- c("-1", "-1", "-1", "", "")
        names(v) <- c("len", "freq", "err", "condition", "pred")
        return(v)
    }
    ys <- target[ixMatch]
    freq <- round(length(ys)/nrow(X), digits = 3)
    if (is.numeric(target)) {
        ysMost <- mean(ys)
        err <- sum((ysMost - ys)^2)/length(ys)
    }
    else {
        if (length(pred) > 0) {
            ysMost = pred
        }
        else {
            ysMost <- names(which.max(table(ys)))
        }
        conf <- round(table(ys)[ysMost]/sum(table(ys)), digits = 3)
        err <- 1 - conf
    }
    rule <- origRule
    v <- c(len, freq, err, rule, ysMost)
    names(v) <- c("len", "freq", "err", "condition", "pred")
    return(v)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
