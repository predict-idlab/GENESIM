\name{singleRuleList2Exec}
\alias{singleRuleList2Exec}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal
}
\description{
internal
}
\usage{
singleRuleList2Exec(ruleList, typeX)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleList}{
%%     ~~Describe \code{ruleList} here~~
}
  \item{typeX}{
%%     ~~Describe \code{typeX} here~~
}
}

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (ruleList, typeX) 
{
    ruleExec <- ""
    vars <- ls(ruleList)
    vars <- vars[order(as.numeric(vars))]
    for (i in 1:length(vars)) {
        if (typeX[as.numeric(vars[i])] == 2) {
            values <- paste("c(", paste(paste("'", ruleList[[vars[i]]], 
                "'", sep = ""), collapse = ","), ")", sep = "")
            tmp = paste("X[,", vars[i], "] \%in\% ", values, sep = "")
        }
        else {
            tmp = ruleList[[vars[i]]]
        }
        if (i == 1) 
            ruleExec <- paste(ruleExec, tmp, sep = "")
        if (i > 1) 
            ruleExec <- paste(ruleExec, " & ", tmp, sep = "")
    }
    return(c(ruleExec))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal}
