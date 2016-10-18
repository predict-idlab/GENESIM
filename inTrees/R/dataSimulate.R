dataSimulate <-
function(flag=1,nCol=20,nRow=1000){
  
  if(nCol<=2) stop("nCol must be >= 2.")
  #only the first and the second features are needed
  X <- matrix(runif(nRow*nCol, min=-2, max=2), ncol=nCol)
  target <- rep(-1,nRow)

  #linear case 
  if(flag == 3) {
    target <- (X[,1]) + (X[,2])
    ix <- which(target>quantile(target, 1/2));
    target <- target*0-1; 
    target[ix] <- 1
  }
  
  #nonlinear case
  if(flag == 2){
    target <- (X[,1])^2 + 1*(X[,2])^2  
    ix <- which(target>quantile(target, 6/10));
    ix <- c(ix,which(target<quantile(target, 1/10)));
    target <- target*0-1; 
    target[ix] <- 1
  }
  
  # team optimization
  if(flag == 1){
    X <- matrix(0,nRow,nCol) 
    for(ii in 1:nrow(X)){ 
      ix <- sample(1:nCol,nCol/2,replace=FALSE)
      X[ii,ix] <- 1
    }
    target <- (xor(X[,1],X[,2]))
    repStr <- function(v){v[v=="1"] <- "Y";v[v=="0"] <- "N";return(v)}
    X <- data.frame(apply(X,2,repStr))
    target[target == FALSE] <- "lose"
    target[target == TRUE] <- "win"
    target <- as.factor(target)

    # X <- data.frame(X)
    # for(jj in 1:ncol(X)){
    #  X[,jj] <- as.factor(X[,jj])
    # }
  }
  return(list(X=X,target=target))
}
