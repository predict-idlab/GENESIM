rfcv <- function(trainx, trainy, cv.fold=5, scale="log", step=0.5,
                 mtry=function(p) max(1, floor(sqrt(p))), recursive=FALSE,
                 ...) {
    classRF <- is.factor(trainy)
    n <- nrow(trainx)
    p <- ncol(trainx)
    if (scale == "log") {
        k <- floor(log(p, base=1/step))
        n.var <- round(p * step^(0:(k-1)))
        same <- diff(n.var) == 0
        if (any(same)) n.var <- n.var[-which(same)]
        if (! 1 %in% n.var) n.var <- c(n.var, 1)
    } else {
        n.var <- seq(from=p, to=1, by=step)
    }
    k <- length(n.var)
    cv.pred <- vector(k, mode="list")
    for (i in 1:k) cv.pred[[i]] <- trainy
    ## Generate the indices of the splits
    ## Stratify the classes for classification problem.
    ## For regression, bin the response into 5 bins and stratify.
    if(classRF) {
        f <- trainy
    } else {
        ##f <- cut(trainy, c(-Inf, quantile(trainy, 1:4/5), Inf))
		f <- factor(rep(1:5, length=length(trainy))[order(order(trainy))])
    }
    nlvl <- table(f)
    idx <- numeric(n)
    for (i in 1:length(nlvl)) {
        idx[which(f == levels(f)[i])] <-  sample(rep(1:cv.fold, length=nlvl[i]))
    }

    for (i in 1:cv.fold) {
        ## cat(".")
        all.rf <- randomForest(trainx[idx != i, , drop=FALSE],
                               trainy[idx != i],
                               trainx[idx == i, , drop=FALSE],
                               trainy[idx == i],
                               mtry=mtry(p), importance=TRUE, ...)
        cv.pred[[1]][idx == i] <- all.rf$test$predicted
        impvar <- (1:p)[order(all.rf$importance[,1], decreasing=TRUE)]
        for (j in 2:k) {
            imp.idx <- impvar[1:n.var[j]]
            sub.rf <-
                randomForest(trainx[idx != i, imp.idx, drop=FALSE],
                             trainy[idx != i],
                             trainx[idx == i, imp.idx, drop=FALSE],
                             trainy[idx == i],
                             mtry=mtry(n.var[j]), importance=recursive, ...)
            cv.pred[[j]][idx == i] <- sub.rf$test$predicted
            ## For recursive selection, use importance measures from the sub-model.
            if (recursive) {
                impvar <-
                    (1:length(imp.idx))[order(sub.rf$importance[,1], decreasing=TRUE)]
      }
      NULL
    }
    NULL
  }
  ## cat("\n")
  if(classRF) {
    error.cv <- sapply(cv.pred, function(x) mean(trainy != x))
  } else {
    error.cv <- sapply(cv.pred, function(x) mean((trainy - x)^2))
  }
  names(error.cv) <- names(cv.pred) <- n.var
  list(n.var=n.var, error.cv=error.cv, predicted=cv.pred)
}
