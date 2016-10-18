\name{extractRules}
\alias{extractRules}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Extract rules from a list of trees
}
\description{
Extract rule conditions from a list of trees. Use functions RF2List/GBM2List to transform RF/GBM objects to list of trees. 
}
\usage{
extractRules(treeList, X, ntree = 100, maxdepth = 6, random = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{treeList}{
 tree list
}
  \item{X}{
%%     ~~Describe \code{X} here~~
predictor variable matrix

}
  \item{ntree}{
conditions are extracted from the first ntree trees 
}
  \item{maxdepth}{
%%     ~~Describe \code{maxdepth} here~~
conditions are extracted from the top maxdepth levels from each tree
}
  \item{random}{
%%     ~~Describe \code{random} here~~
the max depth for each tree is an integer randomly chosen between 1 and maxdepth
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
 a set of rule conditions
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
    data(iris)
    X <- iris[,1:(ncol(iris)-1)]
    target <- iris[,"Species"] 
    rf <- RRF(X,as.factor(target),ntree=100) # build an ordinary RF 
    treeList <- RF2List(rf)
    ruleExec <- extractRules(treeList,X) # transform to R-executable rules
    ruleExec <- unique(ruleExec)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ extract }
