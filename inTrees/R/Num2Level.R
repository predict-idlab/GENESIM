Num2Level <-
function(rfList,splitV){
  for(i in 1:rfList$ntree){ 
    rfList$list[[i]] <- data.frame(rfList$list[[i]])
    rfList$list[[i]][,"prediction"] <- data.frame(dicretizeVector(rfList$list[[i]][,"prediction"],splitV))
    colnames(rfList$list[[i]]) <- c("left daughter","right daughter","split var","split point","status","prediction")
  }
  return(rfList)
}
