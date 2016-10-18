\name{imports85}
\docType{data}
\alias{imports85}
\title{The Automobile Data}
\description{
  This is the `Automobile' data from the UCI Machine Learning Repository.
}
\usage{
data(imports85)
}
\format{
  \code{imports85} is a data frame with 205 cases (rows) and 26
  variables (columns).  This data set consists of three types of
  entities: (a) the specification of an auto in terms of various
  characteristics, (b) its assigned insurance risk rating, (c) its
  normalized losses in use as compared to other cars.  The second rating
  corresponds to the degree to which the auto is more risky than its
  price indicates.  Cars are initially assigned a risk factor symbol
  associated with its price.   Then, if it is more risky (or less), this
  symbol is adjusted by moving it up (or down) the scale.  Actuarians
  call this process `symboling'.  A value of +3 indicates that the auto
  is risky, -3 that it is probably pretty safe.

  The third factor is the relative average loss payment per insured
  vehicle year.  This value is normalized for all autos within a
  particular size classification (two-door small, station wagons,
  sports/speciality, etc...), and represents the average loss per car
  per year.
}
\source{
  Originally created by Jeffrey C. Schlimmer, from 1985 Model Import Car
  and Truck Specifications, 1985 Ward's Automotive Yearbook, Personal
  Auto Manuals, Insurance Services Office, and Insurance Collision
  Report, Insurance Institute for Highway Safety.

  The original data is at \url{http://www.ics.uci.edu/~mlearn/MLSummary.html}.
}
\references{
  1985 Model Import Car and Truck Specifications, 1985 Ward's Automotive
  Yearbook.
  
  Personal Auto Manuals, Insurance Services Office,
  160 Water Street, New York, NY 10038 

  Insurance Collision Report, Insurance Institute for Highway Safety,
  Watergate 600, Washington, DC 20037
}
\seealso{
  \code{\link{randomForest}}
}
\examples{
data(imports85)
imp85 <- imports85[,-2]  # Too many NAs in normalizedLosses.
imp85 <- imp85[complete.cases(imp85), ]
## Drop empty levels for factors.
imp85[] <- lapply(imp85, function(x) if (is.factor(x)) x[, drop=TRUE] else x)

stopifnot(require(randomForest))
price.rf <- randomForest(price ~ ., imp85, do.trace=10, ntree=100)
print(price.rf)
numDoors.rf <- randomForest(numOfDoors ~ ., imp85, do.trace=10, ntree=100)
print(numDoors.rf)
}
\author{Andy Liaw}
\keyword{datasets}