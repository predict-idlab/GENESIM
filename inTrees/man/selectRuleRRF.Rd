\name{selectRuleRRF}
\alias{selectRuleRRF}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
select a set of relevant and non-redundant rules
}
\description{
select a set of relevant and non-redundant rules using regularized random forests
}
\usage{
selectRuleRRF(ruleMetric, X, target)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleMetric}{
a matrix including the rules and metrics
}
  \item{X}{
predictor variable matrix
}
  \item{target}{
%%     ~~Describe \code{target} here~~
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
a matrix including a set of relevant and non-redundant rules, and their metrics
}
%\references{
%Houtao Deng, Interpreting Tree Ensembles with inTrees, technical report, 2014
%}
\author{
Houtao Deng
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{buildLearner}}
}
\examples{
 # See function "buildLearner:
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ select }
