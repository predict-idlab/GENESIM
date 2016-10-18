\name{dicretizeVector}
\alias{dicretizeVector}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
discretize a variable
}
\description{
discretize a variable
}
\usage{
dicretizeVector(v, K = 3)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{v}{ 
 vector
}
  \item{K}{
 discretize into up to K levels with equal frequency
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
 discretized levels for v
}
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
 data(iris)
 dicretizeVector(iris[,1],3)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ discretize }
