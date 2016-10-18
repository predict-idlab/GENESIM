selectRuleRRF <-
function(ruleMetric,X,target){
  ruleI = sapply(ruleMetric[,"condition"],rule2Table,X,target)
  coefReg <- 0.95 - 0.01*as.numeric(ruleMetric[,"len"])/max(as.numeric(ruleMetric[,"len"]))
  rf <- RRF(ruleI,as.factor(target), flagReg = 1, coefReg=coefReg, mtry = (ncol(ruleI)*1/2) , ntree=50, maxnodes= 10,replace=FALSE) 
  imp <- rf$importance/max(rf$importance)
  feaSet <- which(imp > 0.01)
  ruleSetPrunedRRF <- cbind(ruleMetric[feaSet,,drop=FALSE],impRRF=imp[feaSet])
  ix = order(as.numeric(ruleSetPrunedRRF[,"impRRF"]),
              - as.numeric(ruleSetPrunedRRF[,"err"]),
              - as.numeric(ruleSetPrunedRRF[,"len"]),
              decreasing=TRUE)
  ruleSelect <- ruleSetPrunedRRF[ix,,drop=FALSE]
  return(ruleSelect)
}
