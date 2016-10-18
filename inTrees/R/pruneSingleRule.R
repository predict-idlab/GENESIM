pruneSingleRule <-
function(rule, X, target, maxDecay, typeDecay){
# typeDecay = 1: relative error increase; otherwise: absolute error increase

  #A <- gregexpr("X\\[,[0-9]+\\]", s)
  newRuleMetric <- measureRule(rule["condition"],X,target)
  errOrig <- as.numeric(newRuleMetric["err"])  
  ruleV <- unlist(strsplit(rule["condition"],split= " & "))
  pred <- rule["pred"]
  # newRule <- NULL
  if(length(ruleV)==1) return(newRuleMetric)
  for(i in length(ruleV):1){
    restRule <- ruleV[-i]
    restRule <- paste(restRule,collapse= " & ")
    metricTmp <- measureRule(restRule,X,target,pred)
    errNew <- as.numeric(metricTmp["err"]) 
    if(typeDecay == 1){
      decay <- (errNew-errOrig)/max(errOrig,0.000001)
    }else{
      decay <- (errNew-errOrig)
    }      
    if( decay <= maxDecay){
    #if( errNew-errOrig <= maxDecay){
      ruleV <- ruleV[-i] 
      # newRule saves the last changed rule and metrics
      newRuleMetric <- metricTmp
      if(length(ruleV)<=1)break
    }
  }
  return(newRuleMetric)
  #rule["condition"] <- paste(ruleV,collapse= " & ")
  #return(rule)
}
