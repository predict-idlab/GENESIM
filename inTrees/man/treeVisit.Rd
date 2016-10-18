\name{treeVisit}
\alias{treeVisit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
internal function
}
\description{
internal function
}
\usage{
treeVisit(tree, rowIx, count, ruleSet, rule, levelX, length, max_length)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tree}{
%%     ~~Describe \code{tree} here~~
}
  \item{rowIx}{
%%     ~~Describe \code{rowIx} here~~
}
  \item{count}{
%%     ~~Describe \code{count} here~~
}
  \item{ruleSet}{
%%     ~~Describe \code{ruleSet} here~~
}
  \item{rule}{
%%     ~~Describe \code{rule} here~~
}
  \item{levelX}{
%%     ~~Describe \code{levelX} here~~
}
  \item{length}{
%%     ~~Describe \code{length} here~~
}
  \item{max_length}{
%%     ~~Describe \code{max_length} here~~
}
}

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (tree, rowIx, count, ruleSet, rule, levelX, length, 
    max_length) 
{
    if (tree[rowIx, "status"] == -1 | length == max_length) {
        count = count + 1
        ruleSet[[count]] = rule
        return(list(ruleSet = ruleSet, count = count))
    }
    xIx <- tree[rowIx, "split var"]
    xValue <- tree[rowIx, "split point"]
    if (is.null(levelX[[xIx]])) {
        lValue <- paste("X[,", xIx, "]<=", xValue, sep = "")
        rValue <- paste("X[,", xIx, "]>", xValue, sep = "")
    }
    else {
        xValue <- which(as.integer(intToBits(as.integer(xValue))) > 
            0)
        lValue <- levelX[[xIx]][xValue]
        rValue <- setdiff(levelX[[xIx]], lValue)
    }
    xValue <- NULL
    ruleleft <- rule
    if (length(ruleleft) == 0) {
        ruleleft[[as.character(xIx)]] <- lValue
    }
    else {
        if (as.character(xIx) \%in\% ls(ruleleft)) {
            if (!is.null(levelX[[xIx]])) {
                lValue <- intersect(ruleleft[[as.character(xIx)]], 
                  lValue)
                ruleleft[[as.character(xIx)]] <- lValue
            }
            else {
                ruleleft[[as.character(xIx)]] <- paste(ruleleft[[as.character(xIx)]], 
                  "&", lValue)
            }
        }
        else {
            ruleleft[[as.character(xIx)]] <- lValue
        }
    }
    ruleright <- rule
    if (length(ruleright) == 0) {
        ruleright[[as.character(xIx)]] <- rValue
    }
    else {
        if (as.character(xIx) \%in\% ls(ruleright)) {
            if (!is.null(levelX[[xIx]])) {
                rValue <- intersect(ruleright[[as.character(xIx)]], 
                  rValue)
                ruleright[[as.character(xIx)]] <- rValue
            }
            else {
                ruleright[[as.character(xIx)]] <- paste(ruleright[[as.character(xIx)]], 
                  "&", rValue)
            }
        }
        else {
            ruleright[[as.character(xIx)]] <- rValue
        }
    }
    thisList = treeVisit(tree, tree[rowIx, "left daughter"], 
        count, ruleSet, ruleleft, levelX, length + 1, max_length)
    ruleSet = thisList$ruleSet
    count = thisList$count
    thisList = treeVisit(tree, tree[rowIx, "right daughter"], 
        count, ruleSet, ruleright, levelX, length + 1, max_length)
    ruleSet = thisList$ruleSet
    count = thisList$count
    return(list(ruleSet = ruleSet, count = count))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }