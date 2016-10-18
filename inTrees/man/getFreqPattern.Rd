\name{getFreqPattern}
\alias{getFreqPattern}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
calculate frequent variable interactions
}
\description{
calculate frequent variable interactions
}
\usage{
getFreqPattern(ruleMetric, minsup = 0.01, minconf = 0.5, minlen = 1, maxlen = 4)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleMetric}{
%%     ~~Describe \code{ruleMetric} here~~
 a matrix including conditions, predictions, and the metrics
}
  \item{minsup}{
  minimum support of conditions in a tree ensemble
 
}
  \item{minconf}{
%%     ~~Describe \code{conf} here~~
 minimum confidence of the rules
}
  \item{minlen}{
%%     ~~Describe \code{minlen} here~~
 minimum length of the conditions
}
  \item{maxlen}{
%%     ~~Describe \code{maxlen} here~~
 max length of the conditions
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
 a matrix including frequent variable interations (in a form of conditions),  predictions, length, support, and confidence.
}
% \references{
% Houtao Deng, Interpreting Tree Ensembles with Intrees, technical report, 2014
% }
% \author{
% Houtao Deng
% }
% \note{
% %%  ~~further notes~~
% }

% %% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
library(RRF)
library(arules)
data(iris)
X <- iris[,1:(ncol(iris)-1)]
target <- iris[,"Species"] 
rf <- RRF(X,as.factor(target),ntree=100) # build an ordinary RF 
treeList <- RF2List(rf)
ruleExec <- extractRules(treeList,X) # transform to R-executable rules
ruleMetric <- getRuleMetric(ruleExec,X,target) 
freqPattern <- getFreqPattern(ruleMetric)
freqPatternMetric <- getRuleMetric(freqPattern,X,target)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ variable interaction }
