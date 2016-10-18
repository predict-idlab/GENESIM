RF2List <-
function(rf){
  treeList <- NULL
  treeList$ntree <- rf$ntree
  treeList$list <- vector("list",rf$ntree)
  for(i in 1:treeList$ntree){
    treeList$list[[i]] <- getTree(rf,i,labelVar=FALSE)
  }
  return(treeList)
}
