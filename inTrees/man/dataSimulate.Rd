\name{dataSimulate}
\alias{dataSimulate}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate data
}
\description{
Simulate data
}
\usage{
dataSimulate(flag = 1, nCol = 20, nRow = 1000)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{flag}{1 (default): team optimization; 2: non-linear; 3: linear. 
%%     ~~Describe \code{flag} here~~
}
  \item{nCol}{the number of columns in the data set. must >= 2. 
%%     ~~Describe \code{nCol} here~~
}
  \item{nRow}{the number of rows in the data set.  
%%     ~~Describe \code{nRow} here~~
}
}
\value{
    predictor variable matrix and target variable
}
%\references{
%%% ~put references to the literature/web site here ~
%}
%\author{
%%%  ~~who you are~~
%}
%\note{
%%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
res <- dataSimulate(flag=1)
X <- res$X; 
target <- res$target
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ simulate }
