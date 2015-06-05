modelInfo <- list(label = "NORMA with Radial Basis Function Kernel",
                  library = "kernlab",
                  type = c("Classification", "Regression"),
                  parameters = data.frame(parameter = c('sigma', 'nu', 'lambda', 'eta', 'buffersize'), 
                                          class = c("numeric", "numeric", "numeric", "numeric", "integer"),
                                          label = c("Sigma", "Nu", "Regularization", "Learning Rate", "Buffersize")),
                  grid = function(x, y, len = NULL) {
                        library(kernlab)
                        sigmas <- sigest(as.matrix(x), na.action = na.omit, scaled = TRUE)
                        if (length(unique(y)) <= 2)
                            expand.grid(sigma = mean(as.vector(sigmas[-2]))*(c(1,5,10,50,100)),
                                        nu = (1:8)/10, lambda = (1 - 2^(-4:-1)), eta = 10^(-3:-1), buffersize=1000)
                        else
                            expand.grid(sigma = mean(as.vector(sigmas[-2]))*(c(1,5,10,50,100)),
                                        nu = (1:8)/10, lambda = (1 - 2^(-4:-1)), eta = 10^(-3:-1), buffersize=1000)
                  },
                  loop = NULL,
                  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
                    if (length(unique(y)) <= 2)
                        out <- inlearn(d = dim(x)[2], kernel = "rbfdot", kpar = list(param$sigma),
                                       type = "classification", buffersize = param$buffersize)
                    else
                        out <- inlearn(d = dim(x)[2], kernel = "rbfdot", kpar = list(param$sigma),
                                       type = "regression", buffersize = param$buffersize)
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
                    if (length(lev(modelFit)) <= 2) {
                        # two class classification
                        # scale between 0 and 1 with sigmoid, scaling power? use sigma?
                        out <- apply(as.matrix(out), MARGIN = 2, FUN = function(x) 1/(1 + exp(x)))
                        out <- data.frame(out)
                        names(out) <- lev(modelFit)[1]
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
