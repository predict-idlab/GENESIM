\name{getRuleMetric}
\alias{getRuleMetric}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Assign outcomes to a conditions, and measure the rules
}
\description{
Assign outcomes to a conditions, and measure the rules
}
\usage{
getRuleMetric(ruleExec, X, target)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleExec}{
%%     ~~Describe \code{ruleExec} here~~
a set of rule conditions
}
  \item{X}{
%%     ~~Describe \code{X} here~~
predictor variable matrix
}
  \item{target}{
target variable
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
a matrix including the condictions, predictions, and metrics
}
 \references{
 Houtao Deng, Interpreting Tree Ensembles with inTrees, technical report, 2014
 }
% \author{
% Houtao Deng
% }
% \note{
% %%  ~~further notes~~
% }

%% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
library(RRF)
data(iris)
X <- iris[,1:(ncol(iris)-1)]
target <- iris[,"Species"] 
rf <- RRF(X,as.factor(target),ntree=100) # build an ordinary RF 
treeList <- RF2List(rf)
ruleExec <- extractRules(treeList,X) # transform to R-executable rules
ruleExec <- unique(ruleExec)
ruleMetric <- getRuleMetric(ruleExec,X,target) # measure rules
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ measure }
\keyword{ rank }% __ONLY ONE__ keyword per line
