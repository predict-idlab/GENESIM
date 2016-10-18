buildLearner <-
function(ruleMetric,X,target,minFreq=0.01){ #Recursive 
  ruleMetric <- ruleMetric[,c("len","freq","err","condition","pred"),drop=FALSE]
  learner <- NULL
  listIxInst <- vector("list", nrow(ruleMetric))
  for(i in 1:nrow(ruleMetric)){
    ixMatch <- eval(parse(text=paste("which(",ruleMetric[i,"condition"], ")"))  ) 
    if(length(ixMatch)==0)next
    listIxInst[[i]] = ixMatch
  }
  ixInstLeft <- 1:length(target)
  while(TRUE){
    infor = NULL
    restErr <- 1 - max(table(target[ixInstLeft]))/length(target[ixInstLeft])
    for(i in 1:length(listIxInst)){
      thisInfor <- computeRuleInfor(listIxInst[[i]], ruleMetric[i,"pred"],target)
      infor <- rbind(infor,c(thisInfor,len=as.numeric(ruleMetric[i,"len"])))
    }
    topIx <- order(infor[,"err"],-infor[,"freq"],infor[,"len"],decreasing=FALSE)
    minSupIx <- which(infor[,"freq"] < minFreq)
    if(length(minSupIx)>0)topIx <- setdiff(topIx,minSupIx)
    if(length(topIx)>0) topIx <- topIx[1]
    if(length(topIx)==0){
      restCondition <- paste("X[,1]==X[,1]")
      restPred <- names(table(target[ixInstLeft]))[which.max(table(target[ixInstLeft]))]
      restSup <- length(ixInstLeft)/length(target)
      thisRuleMetric <- c(len=1,freq=restSup,err=restErr,condition=restCondition,pred=restPred)
      learner <- rbind(learner,thisRuleMetric)
      break
    }else if( infor[topIx,"err"] >= restErr ){
      restCondition <- paste("X[,1]==X[,1]")
      restPred <- names(table(target[ixInstLeft]))[which.max(table(target[ixInstLeft]))]
      restSup <- length(ixInstLeft)/length(target)
      thisRuleMetric <- c(len=1,freq=restSup,err=restErr,condition=restCondition,pred=restPred)    
      learner <- rbind(learner,thisRuleMetric)
      break      
    }  
    #ruleActiveList <- c(ruleActiveList,topIx)
    thisRuleMetric <- ruleMetric[topIx,,drop=FALSE]
    thisRuleMetric[,c("freq","err","len")] <- infor[topIx,c("freq","err","len")] 
    learner <- rbind(learner,thisRuleMetric)
    ixInstLeft <- setdiff(ixInstLeft,listIxInst[[topIx]])
    listIxInst <- sapply(listIxInst,setdiff,listIxInst[[topIx]])
  
    if(length(ixInstLeft)==0) { # if every is targetified perfectly, still set a main target 
      restCondition <- paste("X[,1]==X[,1]")
      restPred <- names(table(target))[which.max(table(target))]
      restSup <- 0
      restErr <- 0
      thisRuleMetric <- c(len=1,freq=restSup,err=restErr,condition=restCondition,pred=restPred)    
      learner <- rbind(learner,thisRuleMetric)
      break
    }
  }
  rownames(learner) <- NULL
  return(learner)
}
