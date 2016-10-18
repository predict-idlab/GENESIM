\name{presentRules}
\alias{presentRules}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Present a learner using column names instead of X[i,]
}
\description{
Present a learner using column names instead of X[i,]
}
\usage{
presentRules(rules, colN)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{rules}{
%%     ~~Describe \code{rules} here~~
a set of rules
}
  \item{colN}{
a vector including the column names
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
a matrix including the conditions (with column names), etc.
}
% \references{
% Houtao Deng, Interpreting Tree Ensembles with Intrees, technical report, 2014
% }
% \author{
% %%  ~~who you are~~
% }
% \note{
% %%  ~~further notes~~
% }

% %% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{buildLearner}}
}
\examples{
 # See function "buildLearner"
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ present }
