dicretizeVector <-
function(v,K=3){
  splitV <- quantile(v, probs = seq(0, 1, 1/K), na.rm = FALSE,
         names = TRUE, type = 3)
  splitV <- splitV[-c(1,length(splitV))]

  numSplit <- length(splitV)  # split points + 1
  if(numSplit==0) return(v)
  newV <- vector("character", length(v)) 
  newV[which(v<=splitV[1])] = paste("L1",sep="")
  if(numSplit>=2){
    for(jj in 2:numSplit){
      newV[which(  v> splitV[jj-1] & v<=splitV[jj]) ] = paste("L",jj,sep="")     
    }
  }
  newV[which( v> splitV[numSplit] ) ] =  paste("L",(numSplit+1),sep="") 
  return(newV)
}
