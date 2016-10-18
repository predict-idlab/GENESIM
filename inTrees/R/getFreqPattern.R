getFreqPattern <-
function(ruleMetric,minsup=0.01,minconf=0.5,minlen=1,maxlen=4){
  # set up 
  predY <- as.character(ruleMetric[,"pred"])
  rulesV <- strsplit(ruleMetric[,"condition"], split=" & ")
  for(i in 1:length(rulesV)){ 
    rulesV[[i]] = c(rulesV[[i]],paste("=>",predY[i],sep=""))
  }
  yrhs= unique(paste("=>",ruleMetric[,"pred"],sep="")) 
  trans1 <- as(rulesV, "transactions")
  rules1 <- apriori( 
    trans1,
    parameter = list(supp=minsup,conf=minconf,minlen=minlen,maxlen=maxlen), 
    appearance = list(none=NULL,rhs =yrhs,default="lhs")
  )
  #rules1= sort(rules1, decreasing = FALSE, by = "confidence")
  #quality = quality(rules1)
  #qIx = order(quality[,"confidence"],quality[,"support"],decreasing=TRUE)
  #rules1=rules1[qIx]
  #quality = quality[qIx,1:2]
  #inspect(rules1)
  
  lhs = as(lhs(rules1),"list")
  rhs = as(rhs(rules1),"list")
  rhs <- gsub("=>", "", rhs) 
  quality <- quality(rules1)
  ix_empty <- NULL
  freqPattern <- NULL
  for(i in 1:length(lhs)){
    length_v <- length(lhs[[i]])
    lhs[[i]] <- paste(lhs[[i]],collapse= " & ")
    if(nchar(lhs[[i]])==0){
      ix_empty <- c(ix_empty,i)
    }
    freqPattern <- rbind(freqPattern, c(len=length_v, condition=lhs[[i]], pred=rhs[i],
                                        sup=quality[i,"support"], 
                                        conf=quality[i,"confidence"]) )
  }
  if(length(ix_empty)>0)freqPattern <- freqPattern[-ix_empty,]
  qIx = order(as.numeric(freqPattern[,"sup"]), as.numeric(freqPattern[,"conf"]),
              -as.numeric(freqPattern[,"len"]),
              decreasing=TRUE)
  freqPattern <- freqPattern[qIx,c("len","sup","conf","condition","pred")]
  freqPattern[,c("sup","conf")] <- as.character(round(as.numeric(freqPattern[,c("sup","conf")]),digits=3))
  return(freqPattern)
}
