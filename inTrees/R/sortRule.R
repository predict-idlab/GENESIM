sortRule <-
function(M,decreasing=TRUE){
  qIx = order((1- as.numeric(M[,"err"])),
              as.numeric(M[,"freq"]),
              -as.numeric(M[,"len"]),
              decreasing=decreasing)
  return(M[qIx,])
}
