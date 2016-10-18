\name{formatGBM}
\alias{formatGBM}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal
}
\description{
internal
}
\usage{
formatGBM(gbmList, splitBin,X)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gbmList}{
%%     ~~Describe \code{gbmList} here~~
}
  \item{splitBin}{
%%     ~~Describe \code{splitBin} here~~
}
  \item{X}{
 predictor variable matrix
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
function (gbmList, splitBin,X) 
{
    for (j in 1:length(gbmList$list)) {
        a <- gbmList$list[[j]]
        rownames(a) <- 1:nrow(a)
        a$status <- a$SplitVar
        a <- a[, c("LeftNode", "RightNode", "MissingNode", "SplitVar", 
            "SplitCodePred", "status")]
        a[which(a[, "SplitVar"] >= 0), c("SplitVar", "LeftNode", 
            "RightNode", "MissingNode")] <- a[which(a[, "SplitVar"] >= 
            0), c("SplitVar", "LeftNode", "RightNode", "MissingNode")] + 
            1
        ix <- a$MissingNode[which(a$MissingNode > 0)]
        if (length(ix) > 0) 
            a$status[ix] <- 10
        a <- a[, c("LeftNode", "RightNode", "SplitVar", "SplitCodePred", 
            "status")]
        cat <- which(sapply(X, is.factor) & !sapply(X, is.ordered))
        ix <- which(a[, "SplitVar"] \%in\% cat)
        for (i in ix) a[i, "SplitCodePred"] <- splitBin[a[i, 
            "SplitCodePred"] + 1]
        colnames(a) <- c("left daughter", "right daughter", "split var", 
            "split point", "status")
        gbmList$list[[j]] <- a
    }
    return(gbmList)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{internal}
