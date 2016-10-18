treeVisit <-
function(tree,rowIx,count,ruleSet,rule,levelX,length,max_length)
{
  #print(tree[rowIx,"prediction"])
  #print(tree[rowIx,])
  #print("----->")
  #print(tree)
  #print(rowIx)
  if( tree[rowIx,"status"] == -1 | length == max_length ){
    count = count + 1
    ruleSet[[count]] = rule
    return(list(ruleSet = ruleSet, count=count))
  }
  xIx <- tree[rowIx,"split var"]
  xValue <- tree[rowIx,"split point"]
  
  if(is.null(levelX[[xIx]])){
   lValue <- paste("X[,",xIx, "]<=",xValue,sep="")
   rValue <- paste("X[,",xIx, "]>",xValue,sep="")
  }else{
   xValue<- which(as.integer(intToBits(as.integer(xValue)))>0)
   lValue <- levelX[[xIx]][xValue]
   rValue <- setdiff(levelX[[xIx]],lValue)
#   lValue <- paste("X[,",xIx, "]%in% '",lValue,"'",sep="")
#   rValue <- paste("X[,",xIx, "]%in% '",rValue,"'",sep="")
  }  
   xValue <- NULL
   ruleleft <- rule
   if(length(ruleleft)==0)
   {
     ruleleft[[as.character(xIx)]] <- lValue 
   }else{
     if(as.character(xIx) %in% ls(ruleleft)) {
          if(!is.null(levelX[[xIx]])){    
            lValue <- intersect(ruleleft[[as.character(xIx)]],lValue)
            ruleleft[[as.character(xIx)]] <- lValue
          }else{
            ruleleft[[as.character(xIx)]] <- paste(ruleleft[[as.character(xIx)]], "&", lValue)
          }
       }else{
       ruleleft[[as.character(xIx)]] <- lValue
     }
   }
  
   #thisItem = paste("X[,",xIx, "] %in% ", nxValue, sep="")
   ruleright <- rule
   if(length(ruleright)==0)
   {
     ruleright[[as.character(xIx)]] <- rValue
   }else{
     if(as.character(xIx) %in% ls(ruleright)) {
         if(!is.null(levelX[[xIx]])){  
           rValue <- intersect(ruleright[[as.character(xIx)]],rValue)
           ruleright[[as.character(xIx)]] <- rValue
         }else{
           ruleright[[as.character(xIx)]] <- paste(ruleright[[as.character(xIx)]], "&", rValue)
         }
     }else{
        ruleright[[as.character(xIx)]] <- rValue
     }
    }
  
   thisList = treeVisit(tree, tree[rowIx,"left daughter"],count,ruleSet,ruleleft,levelX,length+1,max_length)
   ruleSet = thisList$ruleSet; count = thisList$count
   
   thisList = treeVisit(tree, tree[rowIx,"right daughter"],count,ruleSet,ruleright,levelX,length+1,max_length)
   ruleSet = thisList$ruleSet; count = thisList$count
   
   return(list(ruleSet = ruleSet, count=count))
}
