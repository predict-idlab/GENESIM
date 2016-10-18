getRuleMetric <-
function(ruleExec, X, target){
  #typeX = getTypeX(X)
  #ruleExec <- unique(t(sapply(allRulesList,RuleList2Exec,typeX=typeX)))
  #colnames(ruleExec) <- c("len","condition")
  ruleMetric <- t(sapply(ruleExec[,"condition",drop=FALSE],measureRule,X,target))
  rownames(ruleMetric) = NULL; 
  # ruleMetric <- cbind( ruleExec[,1] ,  ruleMetric )
  colnames(ruleMetric) <- c("len","freq","err","condition","pred")
  dIx <- which(ruleMetric[,"len"]=="-1") 
  if(length(dIx)>0){
   ruleMetric <- ruleMetric[-dIx,]
   print(paste( length(dIx)," paths are ignored.",sep=""))
  }
  return(ruleMetric)
  #qIx = order((1- as.numeric(ruleMetric[,"err"])),
  #            as.numeric(ruleMetric[,"freq"]),
  #            -as.numeric(ruleMetric[,"len"]),
  #            decreasing=TRUE)
  #return(ruleMetric[qIx,])
}
