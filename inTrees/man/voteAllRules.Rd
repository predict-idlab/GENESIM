\name{voteAllRules}
\alias{voteAllRules}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal
}
\description{
Predictions from a rule set
}
\usage{
voteAllRules(ruleMetric, X, type = "r", method = "median")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleMetric}{
rules and metrics
}
  \item{X}{
predictor variable matrix
}
  \item{type}{
regression or classification
}
  \item{method}{
for regression, use median or average
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
predictions from the rule set
}
% \references{
% %% ~put references to the literature/web site here ~
% }
% \author{
% %%  ~~who you are~~
% }
% \note{
% %%  ~~further notes~~
% }

%% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (ruleMetric, X, type = "r", method = "median") 
{
    xVoteList = vector("list", nrow(X))
    predY <- rep("", nrow(X))
    for (i in 1:nrow(ruleMetric)) {
        ixMatch <- eval(parse(text = paste("which(", ruleMetric[i, 
            "condition"], ")")))
        if (length(ixMatch) == 0) 
            next
        for (ii in ixMatch) {
            xVoteList[[ii]] = c(xVoteList[[ii]], ruleMetric[i, 
                "pred"])
        }
    }
    for (i in 1:length(xVoteList)) {
        thisV <- xVoteList[[i]]
        if (length(thisV) == 0) 
            next
        if (type == "c") 
            predY[i] <- names(table(thisV)[which.max(table(thisV))])
        if (type == "r") {
            thisV = as.numeric(thisV)
            if (method == "median") {
                predY[i] <- median(thisV)
            }
            else {
                predY[i] <- mean(thisV)
            }
        }
    }
    if (type == "r") 
        predY <- as.numeric(predY)
    return(predY)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }

