GBM2List <-
function(gbm1,X){
  treeList <- NULL
  treeList$ntree <- gbm1$n.trees
  treeList$list <- vector("list",gbm1$n.trees)
  for(i in 1:treeList$ntree){
    treeList$list[[i]] <- pretty.gbm.tree(gbm1,i.tree = i)
  }

  v2int <- function(v){sum( (-v+1)/2 * 2^seq(0,(length(v)-1),1)  )}
  #as.integer(intToBits(3616)) pretty.gbm.tree(gbm1,i.tree = 1)
  splitBin = sapply(gbm1$c.splits,v2int)

  return(formatGBM(treeList,splitBin,X))
}
