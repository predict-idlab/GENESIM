\name{buildLearner}
\alias{buildLearner}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
build a simplified tree ensemble learner (STEL)
}
\description{
Build a simplified tree ensemble learner (STEL). Currently works only for classification problems. 
}
\usage{
buildLearner(ruleMetric, X, target, minFreq = 0.01)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleMetric}{
a matrix including the conditions, predictions, and and metrics
}
  \item{X}{
predictor variable matrix
}
  \item{target}{
target variable
}
  \item{minFreq}{
minimum frequency of a rule condition in order to be included in STEL. 
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
	a matrix including the conditions, prediction, and metrics, ordered by priority.
}
 \references{
 Houtao Deng, Interpreting Tree Ensembles with inTrees, technical report, 2014
 }
 \author{
 Houtao Deng
 }
% \note{
% %%  ~~further notes~~
% }

%% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
data(iris)
library(RRF)
X <- iris[,1:(ncol(iris)-1)]
target <- iris[,"Species"] 
rf <- RRF(X,as.factor(target),ntree=100) # build an ordinary RF 
treeList <- RF2List(rf)
ruleExec <- extractRules(treeList,X)
ruleExec <- unique(ruleExec)
ruleMetric <- getRuleMetric(ruleExec,X,target) # measure rules
ruleMetric <- pruneRule(ruleMetric,X,target) # prune each rule
#ruleMetric <- selectRuleRRF(ruleMetric,X,target) # rule selection
learner <- buildLearner(ruleMetric,X,target)
pred <- applyLearner(learner,X)
read <- presentRules(learner,colnames(X)) # more readable format

# format the rule and metrics as a table in latex code
library(xtable)
print(xtable(read), include.rownames=FALSE)
print(xtable(ruleMetric[1:2,]), include.rownames=FALSE)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ STEL }
\keyword{ learner }% __ONLY ONE__ keyword per line
