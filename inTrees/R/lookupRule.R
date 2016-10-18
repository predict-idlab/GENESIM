lookupRule <-
function(rules,strList){
  ix <- grep(strList[1], rules[,"condition"],fixed = TRUE)
  if(length(strList)>=2){
    for(i in 2:length(strList)){
      ix2 <- grep(strList[i], rules[,"condition"],fixed = TRUE)
      ix <- intersect(ix,ix2)
    }
  }
  if(length(ix)>=1)return(rules[ix,,drop=FALSE])
  if(length(ix)==0)return(NULL) 
}
