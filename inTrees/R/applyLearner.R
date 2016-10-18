applyLearner <-
function(learner,X){
  leftIx <- 1:nrow(X)
  predY <- rep("",nrow(X))
  for(i in 1:nrow(learner)){
    ixMatch <- eval(parse(text=paste("which(",learner[i,"condition"], ")"))  ) 
    ixMatch <- intersect(leftIx,ixMatch)
    if(length(ixMatch)>0){
      predY[ixMatch] <- learner[i,"pred"]
      leftIx <- setdiff(leftIx,ixMatch)
    }
    if(length(leftIx)==0){
      break
    }
  }
  return(predY)
}
