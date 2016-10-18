\name{pruneSingleRule}
\alias{pruneSingleRule}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal
}
\description{
internal
}
\usage{
pruneSingleRule(rule, X, target, maxDecay, typeDecay)
}
% %- maybe also 'usage' for other objects documented here.
% \arguments{
%   \item{rule}{
% %%     ~~Describe \code{rule} here~~
% }
%   \item{X}{
% %%     ~~Describe \code{X} here~~
% }
%   \item{target}{
% %%     ~~Describe \code{target} here~~
% }
%   \item{maxDecay}{
% %%     ~~Describe \code{maxDecay} here~~
% }
%   \item{typeDecay}{
% %%     ~~Describe \code{typeDecay} here~~
% }
% }
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
% \value{
% %%  ~Describe the value returned
% %%  If it is a LIST, use
% %%  \item{comp1 }{Description of 'comp1'}
% %%  \item{comp2 }{Description of 'comp2'}
% %% ...
% }
% \references{
% %% ~put references to the literature/web site here ~
% }
% \author{
% %%  ~~who you are~~
% }
% \note{
% %%  ~~further notes~~
% }

% %% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (rule, X, target, maxDecay, typeDecay) 
{
    newRuleMetric <- measureRule(rule["condition"], X, target)
    errOrig <- as.numeric(newRuleMetric["err"])
    ruleV <- unlist(strsplit(rule["condition"], split = " & "))
    pred <- rule["pred"]
    if (length(ruleV) == 1) 
        return(newRuleMetric)
    for (i in length(ruleV):1) {
        restRule <- ruleV[-i]
        restRule <- paste(restRule, collapse = " & ")
        metricTmp <- measureRule(restRule, X, target, pred)
        errNew <- as.numeric(metricTmp["err"])
        if (typeDecay == 1) {
            decay <- (errNew - errOrig)/max(errOrig, 1e-06)
        }
        else {
            decay <- (errNew - errOrig)
        }
        if (decay <= maxDecay) {
            ruleV <- ruleV[-i]
            newRuleMetric <- metricTmp
            if (length(ruleV) <= 1) 
                break
        }
    }
    return(newRuleMetric)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
