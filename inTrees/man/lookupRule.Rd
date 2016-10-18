\name{lookupRule}
\alias{lookupRule}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal
}
\description{
internal
}
\usage{
lookupRule(rules, strList)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{rules}{
%%     ~~Describe \code{rules} here~~
}
  \item{strList}{
%%     ~~Describe \code{strList} here~~
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

% %% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (rules, strList) 
{
    ix <- grep(strList[1], rules[, "condition"])
    if (length(strList) >= 2) {
        for (i in 2:length(strList)) {
            ix2 <- grep(strList[i], rules[, "condition"])
            ix <- intersect(ix, ix2)
        }
    }
    if (length(ix) >= 1) 
        return(rules[ix, , drop = FALSE])
    if (length(ix) == 0) 
        return(NULL)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
