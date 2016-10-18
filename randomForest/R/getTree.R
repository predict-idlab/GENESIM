getTree <- function(rfobj, k=1, labelVar=FALSE) {
  if (is.null(rfobj$forest)) {
    stop("No forest component in ", deparse(substitute(rfobj)))
  }
  if (k > rfobj$ntree) {
    stop("There are fewer than ", k, "trees in the forest")
  }
  if (rfobj$type == "regression") {
      tree <- cbind(rfobj$forest$leftDaughter[,k],
                    rfobj$forest$rightDaughter[,k],
                    rfobj$forest$bestvar[,k],
                    rfobj$forest$xbestsplit[,k],
                    rfobj$forest$nodestatus[,k],
                    rfobj$forest$nodepred[,k])[1:rfobj$forest$ndbigtree[k],]
  } else {
      tree <- cbind(rfobj$forest$treemap[,,k],
                    rfobj$forest$bestvar[,k],
                    rfobj$forest$xbestsplit[,k],
                    rfobj$forest$nodestatus[,k],
                    rfobj$forest$nodepred[,k])[1:rfobj$forest$ndbigtree[k],]
  }

  dimnames(tree) <- list(1:nrow(tree), c("left daughter", "right daughter",
                                         "split var", "split point",
                                         "status", "prediction"))

  if (labelVar) {
      tree <- as.data.frame(tree)
      v <- tree[[3]]
      v[v == 0] <- NA
      tree[[3]] <- factor(rownames(rfobj$importance)[v])
      if (rfobj$type == "classification") {
          v <- tree[[6]]
          v[! v %in% 1:nlevels(rfobj$y)] <- NA
          tree[[6]] <- levels(rfobj$y)[v]
      }
  }
  tree
}

