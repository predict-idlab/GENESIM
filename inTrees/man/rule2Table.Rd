\name{rule2Table}
\alias{rule2Table}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal function
}
\description{
internal function
}
\usage{
rule2Table(ruleExec, X, target)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ruleExec}{
%%     ~~Describe \code{ruleExec} here~~
}
  \item{X}{
%%     ~~Describe \code{X} here~~
}
  \item{target}{
%%     ~~Describe \code{class} here~~
}
}
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

%% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (ruleExec, X, target) 
{
    I <- rep(0, nrow(X))
    ruleExec <- paste("which(", ruleExec, ")")
    ixMatch <- eval(parse(text = ruleExec))
    if (length(ixMatch) > 0) 
        I[ixMatch] <- 1
    names(I) = NULL
    return(I)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{internal}
