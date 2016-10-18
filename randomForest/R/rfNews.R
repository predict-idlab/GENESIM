rfNews <- function() {
    newsfile <- file.path(system.file(package="randomForest"), "NEWS")
    file.show(newsfile)
}
