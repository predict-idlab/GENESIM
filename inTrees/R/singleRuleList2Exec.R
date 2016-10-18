singleRuleList2Exec <-
function(ruleList,typeX){ #numeric: 1; categorical: 2s
  #ruleExec <- "which("
  ruleExec <- ""
  vars <- ls(ruleList)
  #ruleL <- length(unique(vars))
  vars <- vars[order(as.numeric(vars))]
  for(i in 1:length(vars)){
    if(typeX[as.numeric(vars[i])]==2){
      values <- paste("c(",paste(  paste("'",ruleList[[vars[i]]],"'",sep="")    ,collapse=","),")",sep="")
      tmp = paste("X[,",vars[i], "] %in% ", values, sep="")
    }else{
      tmp = ruleList[[vars[i]]]
    }
    if(i==1)ruleExec <- paste(ruleExec, tmp,sep="")
    if(i>1)ruleExec <- paste(ruleExec, " & ", tmp, sep="")
  }
  #ruleExec <- paste(ruleExec,")",sep="")  
  return(c(ruleExec))
}
