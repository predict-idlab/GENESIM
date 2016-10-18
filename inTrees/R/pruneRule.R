pruneRule <-
function(rules,X,target, maxDecay = 0.05, typeDecay = 2){
  newRuleMetric <- NULL
  for(i in 1:nrow(rules)){
    newRuleMetric <- rbind(newRuleMetric, pruneSingleRule(rules[i,],X,target, maxDecay, typeDecay))
  }
  return(newRuleMetric)
}
