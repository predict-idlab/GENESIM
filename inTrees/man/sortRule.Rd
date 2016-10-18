\name{sortRule}
\alias{sortRule}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal 
}
\description{
internal
}
\usage{
sortRule(M, decreasing = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{M}{
%%     ~~Describe \code{M} here~~
}
  \item{decreasing}{
%%     ~~Describe \code{decreasing} here~~
}
}

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (M, decreasing = TRUE) 
{
    qIx = order((1 - as.numeric(ruleMetric[, "err"])), as.numeric(ruleMetric[, 
        "freq"]), -as.numeric(ruleMetric[, "len"]), decreasing = decreasing)
    return(M[qIx, ])
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
