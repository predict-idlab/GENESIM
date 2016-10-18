\name{GBM2List}
\alias{GBM2List}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Transform gbm object to a list of trees
}
\description{
Transform gbm object to a list of trees that can be used for rule condition extraction
}
\usage{
GBM2List(gbm1,X)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gbm1}{
%%     ~~Describe \code{gbm1} here~~
gbm object
}
  \item{X}{
predictor variable matrix
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
  a list of trees in an inTrees-required format
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
\code{\link{RF2List}}
}
\examples{
    library(gbm)
    data(iris)
    X <- iris[,1:(ncol(iris)-1)]
    target <- iris[,"Species"] 
    gbmFit <- gbm(Species~ ., data=iris, n.tree = 400,
                    interaction.depth = 10,distribution="multinomial")
    treeList <- GBM2List(gbmFit,X)
    ruleExec = extractRules(treeList,X)
    ruleExec <- unique(ruleExec)
    #ruleExec <- ruleExec[1:min(2000,length(ruleExec)),,drop=FALSE]
    ruleMetric <- getRuleMetric(ruleExec,X,target)
    ruleMetric <- pruneRule(ruleMetric,X,target)
    ruleMetric <- unique(ruleMetric)
    learner <- buildLearner(ruleMetric,X,target)
    pred <- applyLearner(learner,X)
    readableLearner <- presentRules(learner,colnames(X)) # more readable format
    err <- 1-sum(pred==target)/length(pred);
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ gbm }
