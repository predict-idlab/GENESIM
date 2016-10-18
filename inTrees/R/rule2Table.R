rule2Table <-
function(ruleExec,X,target){
  I <- rep(0,nrow(X))
  ruleExec <- paste("which(", ruleExec, ")")
  ixMatch <- eval(parse(text=ruleExec)) 
  if(length(ixMatch)>0) I[ixMatch] <- 1
  names(I) = NULL
  return(I)
}
