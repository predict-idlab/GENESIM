combine <- function(...) {
   pad0 <- function(x, len) c(x, rep(0, len-length(x)))
   padm0 <- function(x, len) rbind(x, matrix(0, nrow=len-nrow(x),
                                             ncol=ncol(x)))
   rflist <- list(...)
   areForest <- sapply(rflist, function(x) inherits(x, "randomForest")) 
   if (any(!areForest)) stop("Argument must be a list of randomForest objects")
   ## Use the first component as a template
   rf <- rflist[[1]]
   classRF <- rf$type == "classification"
   trees <- sapply(rflist, function(x) x$ntree)
   ntree <- sum(trees)
   rf$ntree <- ntree
   nforest <- length(rflist)
   haveTest <- ! any(sapply(rflist, function(x) is.null(x$test)))
   ## Check if predictor variables are identical.
   vlist <- lapply(rflist, function(x) rownames(importance(x)))
   numvars <- sapply(vlist, length)
   if (! all(numvars[1] == numvars[-1]))
       stop("Unequal number of predictor variables in the randomForest objects.")
   for (i in seq_along(vlist)) {
       if (! all(vlist[[i]] == vlist[[1]]))
           stop("Predictor variables are different in the randomForest objects.")
   }
   ## Combine the forest component, if any
   haveForest <- sapply(rflist, function(x) !is.null(x$forest))
   if (all(haveForest)) {
       nrnodes <- max(sapply(rflist, function(x) x$forest$nrnodes))
       rf$forest$nrnodes <- nrnodes
       rf$forest$ndbigtree <-
           unlist(sapply(rflist, function(x) x$forest$ndbigtree))
       rf$forest$nodestatus <-
           do.call("cbind", lapply(rflist, function(x)
                                   padm0(x$forest$nodestatus, nrnodes)))
       rf$forest $bestvar <-
           do.call("cbind",
                   lapply(rflist, function(x)
                          padm0(x$forest$bestvar, nrnodes)))
       rf$forest$xbestsplit <-
           do.call("cbind",
                   lapply(rflist, function(x)
                          padm0(x$forest$xbestsplit, nrnodes)))
       rf$forest$nodepred <-
           do.call("cbind", lapply(rflist, function(x)
                                   padm0(x$forest$nodepred, nrnodes)))
       tree.dim <- dim(rf$forest$treemap)
       if (classRF) {
           rf$forest$treemap <-
               array(unlist(lapply(rflist, function(x) apply(x$forest$treemap, 2:3,
                                                             pad0, nrnodes))),
                     c(nrnodes, 2, ntree))
       } else {
           rf$forest$leftDaughter <-
               do.call("cbind",
                       lapply(rflist, function(x)
                          padm0(x$forest$leftDaughter, nrnodes)))
           rf$forest$rightDaughter <-
               do.call("cbind",
                                  lapply(rflist, function(x)
                          padm0(x$forest$rightDaughter, nrnodes)))
       }
           rf$forest$ntree <- ntree
       if (classRF) rf$forest$cutoff <- rflist[[1]]$forest$cutoff
   } else {
       rf$forest <- NULL
   }
   
   if (classRF) {
       ## Combine the votes matrix: 
       rf$votes <- 0
       rf$oob.times <- 0
       areVotes <- all(sapply(rflist, function(x) any(x$votes > 1, na.rf=TRUE)))
       if (areVotes) {
           for(i in 1:nforest) {
               rf$oob.times <- rf$oob.times + rflist[[i]]$oob.times
               rf$votes <- rf$votes +
                   ifelse(is.na(rflist[[i]]$votes), 0, rflist[[i]]$votes)
           }
       } else {
           for(i in 1:nforest) {
               rf$oob.times <- rf$oob.times + rflist[[i]]$oob.times            
               rf$votes <- rf$votes +
                   ifelse(is.na(rflist[[i]]$votes), 0, rflist[[i]]$votes) *
                       rflist[[i]]$oob.times
           }
           rf$votes <- rf$votes / rf$oob.times
       }
       rf$predicted <- factor(colnames(rf$votes)[max.col(rf$votes)],
                              levels=levels(rf$predicted))
       if(haveTest) {
           rf$test$votes <- 0
           if (any(rf$test$votes > 1)) {
               for(i in 1:nforest)
                   rf$test$votes <- rf$test$votes + rflist[[i]]$test$votes
           } else {
               for (i in 1:nforest)
                   rf$test$votes <- rf$test$votes +
                       rflist[[i]]$test$votes * rflist[[i]]$ntree
           }
           rf$test$predicted <-
               factor(colnames(rf$test$votes)[max.col(rf$test$votes)],
                      levels=levels(rf$test$predicted))
       }
   } else {
       rf$predicted <- 0
       for (i in 1:nforest) rf$predicted <- rf$predicted +
           rflist[[i]]$predicted * rflist[[i]]$ntree
       rf$predicted <- rf$predicted / ntree
       if (haveTest) {
           rf$test$predicted <- 0
           for (i in 1:nforest) rf$test$predicted <- rf$test$predicted +
               rflist[[i]]$test$predicted * rflist[[i]]$ntree
           rf$test$predicted <- rf$test$predicted / ntree
       }
   }
   
   ## If variable importance is in all of them, compute the average
   ## (weighted by the number of trees in each forest)
   have.imp <- !any(sapply(rflist, function(x) is.null(x$importance)))
   if (have.imp) {
       rf$importance <- rf$importanceSD <- 0
       for(i in 1:nforest) {
           rf$importance <- rf$importance +
               rflist[[i]]$importance * rflist[[i]]$ntree
           ## Do the same thing with SD of importance, though that's not
           ## exactly right...
           rf$importanceSD <- rf$importanceSD +
               rflist[[i]]$importanceSD^2 * rflist[[i]]$ntree
       }
       rf$importance <- rf$importance / ntree
       rf$importanceSD <- sqrt(rf$importanceSD / ntree)
       haveCaseImp <- !any(sapply(rflist, function(x)
                                  is.null(x$localImportance)))
       ## Average casewise importance
       if (haveCaseImp) {
           rf$localImportance <- 0
           for (i in 1:nforest) {
               rf$localImportance <- rf$localImportance +
                   rflist[[i]]$localImportance * rflist[[i]]$ntree
           }
           rf$localImportance <- rf$localImportance / ntree
       }
   }
   
   ## If proximity is in all of them, compute the average
   ## (weighted by the number of trees in each forest)
   have.prox <- !any(sapply(rflist, function(x) is.null(x$proximity)))
   if (have.prox) {
       rf$proximity <- 0
       for(i in 1:nforest)
           rf$proximity <- rf$proximity + rflist[[i]]$proximity * rflist[[i]]$ntree
       rf$proximity <- rf$proximity / ntree
   }
   	
	## if there are inbag matrices, combine them as well.
	hasInBag <- all(sapply(rflist, function(x) !is.null(x$inbag)))
   	if (hasInBag) rf$inbag <- do.call(cbind, lapply(rflist, "[[", "inbag"))
   	## Set confusion matrix and error rates to NULL
   	if (classRF) {
       	rf$confusion <- NULL
       	rf$err.rate <- NULL
       	if (haveTest) {
           	rf$test$confusion <- NULL
           	rf$err.rate <- NULL
       	}
   	} else {
    	rf$mse <- rf$rsq <- NULL
       	if (haveTest) rf$test$mse <- rf$test$rsq <- NULL
   	}   
   	rf
}
