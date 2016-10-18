formatGBM <-
function(gbmList,splitBin,X){
  for(j in 1:length(gbmList$list)){
  a <- gbmList$list[[j]]
  rownames(a) <- 1:nrow(a)
  a$status <- a$SplitVar
  a <- a[,c("LeftNode","RightNode","MissingNode","SplitVar","SplitCodePred","status")]
  a[which(a[,"SplitVar"]>=0),c("SplitVar","LeftNode","RightNode","MissingNode")] <- a[which(a[,"SplitVar"]>=0),c("SplitVar","LeftNode","RightNode","MissingNode")] + 1
  ix <- a$MissingNode[which(a$MissingNode>0)]
  if(length(ix)>0)  a$status[ix] <- 10 #missing #a <- a[-ix,]
  a <- a[,c("LeftNode","RightNode","SplitVar","SplitCodePred","status")]
  cat <- which(sapply(X, is.factor) & !sapply(X, is.ordered))
  ix <- which(a[,"SplitVar"] %in% cat)
  
  for(i in ix) a[i,"SplitCodePred"] <- splitBin[ a[i,"SplitCodePred"]+1 ]
  colnames(a) <- c("left daughter","right daughter","split var","split point","status")
  gbmList$list[[j]] <- a
  }
  return(gbmList)
}
