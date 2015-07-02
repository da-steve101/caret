modelInfo <- list(label = "NORMA with Radial Basis Function Kernel",
                  library = "kernlab",
                  type = c("Classification", "Regression"),
                  parameters = data.frame(parameter = c('sigma', 'nu', 'lambda', 'eta', 'buffersize', 'normaType', 'kernel'), 
                                          class = c("numeric", "numeric", "numeric", "numeric", "integer", "character", "character"),
                                          label = c("Sigma", "Nu", "Regularization", "Learning Rate", "Buffersize", "Type", "Kernel Function")),
                  grid = function(x, y, len = NULL) {
                        library(kernlab)
                        sigmas <- sigest(as.matrix(x), na.action = na.omit, scaled = TRUE)
                        normaGrid<-expand.grid(sigma = mean(as.vector(sigmas[-2]))*(c(1,5,10,50,100)),                                               nu = (1:8)/10, lambda = (1 - 2^(-4:-1)),
                                               eta = 10^(-3:-1), buffersize=1000)
                        normaGrid$kernel <- "rbfdot"
                        if (length(unique(y)) <= 1)
                            normaGrid$normaType <- "novelty"
                        else if (length(unique(y)) <= 2)
                            normaGrid$normaType <- "classification"
                        else
                            normaGrid$normaType <- "regression"
                  },
                  loop = NULL,
                  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
                    out <- inlearn(d = dim(x)[2], kernel = param$kernel, kpar = list(param$sigma),
                                   type = param$normaType, buffersize = param$buffersize)
                    out <- onlearn(out, x = as.matrix(x), y = y, nu = param$nu, lambda = param$lambda, eta = param$eta)
                    out
                    },
                  predict = function(modelFit, newdata, submodels = NULL) {
                    NORMAPred <- function(obj, x)
                    {
                      pred <- predict(obj, x)
                      if (length(lev(obj)) == 2) {
                          # two class classification      
                          pred <- factor((pred > 0)*1 + 1)
                          levels(pred) <- lev(obj)
                      }
                      pred
                    }
                    out <- try(NORMAPred(modelFit, newdata), silent = FALSE)
                    out
                  },
                  prob = function(modelFit, newdata, submodels = NULL) {
                    NORMAPred <- function(obj, x)
                    {
                      pred <- predict(obj, x)
                      pred
                    }
                    out <- try(NORMAPred(modelFit, newdata), silent = FALSE)
                    if (class(out)[1] != "try-error") {
                      if (length(lev(modelFit)) <= 2) {
                        # two class classification
                        # scale between 0 and 1 with sigmoid, scaling power? use sigma?
                        out <- apply(as.matrix(out), MARGIN = 2, FUN = function(x) 1/(1 + exp(-x)))
                        out <- data.frame(out, 1 - out)
                        names(out) <- lev(modelFit)
                      }
                    } else {
                        warning("NORMA prediction try error")
                        out <- matrix(data=NA,nrow=1,ncol=dim(newdata)[2])
                    }
                    out
                  },
                  predictors = function(x, ...){
                    if(hasTerms(x) & !is.null(x@terms))
                    {
                      out <- predictors.terms(x@terms)
                    } else {
                      out <- colnames(attr(x, "xmatrix"))
                    }
                    if(is.null(out)) out <- names(attr(x, "scaling")$x.scale$`scaled:center`)
                    if(is.null(out)) out <-NA
                    out
                  },
                  tags = c("Kernel Method", "Online Learning", "Radial Basis Function"),
                  levels = function(x) lev(x),
                  sort = function(x) {
                    x[order(x$nu),]
                  })
