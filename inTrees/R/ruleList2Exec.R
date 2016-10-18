ruleList2Exec <-
function(X,allRulesList){
  typeX = getTypeX(X)
  ruleExec <- unique(t(sapply(allRulesList,singleRuleList2Exec,typeX=typeX)))
  ruleExec <- t(ruleExec)
  colnames(ruleExec) <- "condition"
  return(ruleExec)
}
