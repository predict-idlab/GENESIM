\name{applyLearner}
\alias{applyLearner}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
apply a simplified tree ensemble learner (STEL) to data
}
\description{
apply STEL to data and get predictions
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
applyLearner(learner, X)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{learner}{
  a matrix with rules ordered by priority
}
  \item{X}{
  predictor variable matrix
}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
 predictions for the data
}
% \references{
% Houtao Deng, Interpreting Tree Ensembles with Intrees, technical report, 2014
% }
% \author{
% Houtao Deng
% }
% \note{
% %%  ~~further notes~~
% }

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{buildLearner}}
}
\examples{
# see function "buildLearner" for examples
# pred <- applyLearner(learner,X)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ apply }
\keyword{ predict }% __ONLY ONE__ keyword per line
