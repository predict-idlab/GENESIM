\name{pruneRule}
\alias{pruneRule}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prune irrevant variable-value pair from a rule condition
}
\description{
Prune irrevant variable-value pair from a rule condition
}
\usage{
pruneRule(rules, X, target, maxDecay = 0.05, typeDecay = 2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{rules}{
A metrix including the rules and metrics
}
  \item{X}{
predictor variable matrix
}
  \item{target}{
target variable vector
}
  \item{maxDecay}{
threshold of decay
}
  \item{typeDecay}{
1: relative error; 2: error; default :2
}
}

\value{
A matrix including the rules each being pruned, and metrics
}
\references{
 Houtao Deng, Interpreting Tree Ensembles with inTrees, technical report, 2014
}

\author{
Houtao Deng
}

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{buildLearner}}
}
\examples{
# see function "buildLearner"
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ prune }
