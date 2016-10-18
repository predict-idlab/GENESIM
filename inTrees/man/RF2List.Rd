\name{RF2List}
\alias{RF2List}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Transform a random forest object to a list of trees
}
\description{
Transform a random forest object to a list of trees
}
\usage{
RF2List(rf)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{rf}{
%%     ~~Describe \code{rf} here~~
random forest object
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
a list of trees 
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

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{GBM2List}}
}
\examples{
library(RRF)
data(iris)
X <- iris[,1:(ncol(iris)-1)]
target <- iris[,"Species"] 
rf <- RRF(X,as.factor(target),ntree=100) # build an ordinary RF 
treeList <- RF2List(rf)
ruleExec <- extractRules(treeList,X) # transform to R-executable rules
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ randomforest }
