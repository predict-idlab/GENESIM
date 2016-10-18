computeRuleInfor <-
function(instIx,pred,target){
  trueCls <- as.character(target[instIx])
  err <- 1- length(which(trueCls == pred))/length(trueCls)
  return(c(err=err,freq=length(instIx)/length(target)))
}
