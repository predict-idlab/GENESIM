classCenter <- function(x, label, prox, nNbr = min(table(label))-1) {
    ## nPrototype=rep(3, length(unique(label))), ...) {
    label <- as.character(label)
    clsLabel <- unique(label)
    ## Find the nearest nNbr neighbors of each case
    ## (including the case itself). 
    idx <- t(apply(prox, 1, order, decreasing=TRUE)[1:nNbr,])
    ## Find the class labels of the neighbors.
    cls <- label[idx]
    dim(cls) <- dim(idx)
    ## Count the number of neighbors in each class for each case.
    ncls <- sapply(clsLabel, function(x) rowSums(cls == x))
    ## For each class, find the case(s) with most neighbors in that class.
    clsMode <- max.col(t(ncls)) 
    ## Identify the neighbors of the class modes that are of the target class.
    nbrList <- mapply(function(cls, m) idx[m,][label[idx[m,]] == cls],
                      clsLabel, clsMode, SIMPLIFY=FALSE)
    ## Get the X data for the neighbors of the class `modes'.
    xdat <- t(sapply(nbrList, function(i) apply(x[i,,drop=FALSE], 2,
                                                  median)))
    xdat
}    
