varImpPlot <- function(x, sort=TRUE,
                       n.var=min(30, nrow(x$importance)),
                       type=NULL, class=NULL, scale=TRUE, 
                       main=deparse(substitute(x)), ...) {
    if (!inherits(x, "randomForest"))
        stop("This function only works for objects of class `randomForest'")
    imp <- importance(x, class=class, scale=scale, type=type, ...)
    ## If there are more than two columns, just use the last two columns.
    if (ncol(imp) > 2) imp <- imp[, -(1:(ncol(imp) - 2))]
    nmeas <- ncol(imp)
    if (nmeas > 1) {
        op <- par(mfrow=c(1, 2), mar=c(4, 5, 4, 1), mgp=c(2, .8, 0),
                  oma=c(0, 0, 2, 0), no.readonly=TRUE)
        on.exit(par(op))
    }
    for (i in 1:nmeas) {
        ord <- if (sort) rev(order(imp[,i],
                                   decreasing=TRUE)[1:n.var]) else 1:n.var
        xmin <- if (colnames(imp)[i] %in%
                    c("IncNodePurity", "MeanDecreaseGini")) 0 else min(imp[ord, i])
        dotchart(imp[ord,i], xlab=colnames(imp)[i], ylab="",
                 main=if (nmeas == 1) main else NULL,
                 xlim=c(xmin, max(imp[,i])), ...)
    }
    if (nmeas > 1) mtext(outer=TRUE, side=3, text=main, cex=1.2)
    invisible(imp)
}
