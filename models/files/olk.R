modelInfo <- list(label = "OLK with Radial Basis Function Kernel",
                  library = "kernlab",
                  type = c("Classification", "Regression"),
                  parameters = data.frame(parameter = c('sigma', 'C', 'r', 'epsilon', 'buffersize'),
                                          class = c("numeric", "numeric", "numeric", "numeric", "integer"),
                                          label = c("Sigma", "Cost", "Forgetting factor", "Regression Error Tol", "Buffersize")),
                  grid = function(x, y, len = NULL) {
                        library(kernlab)
                        sigmas <- sigest(as.matrix(x), na.action = na.omit, scaled = TRUE)
                        if (length(unique(y)) <= 2)
                            expand.grid(sigma = mean(as.vector(sigmas[-2])),
                                        C = 2 ^((1:len) - 3), r = 10^(-5:-1), epsilon=0, buffersize=1000)
                        else
                            expand.grid(sigma = mean(as.vector(sigmas[-2])),
                                        C = 2 ^((1:len) - 3), r = 10^(-5:-1), epsilon= 10^(-3:-1), buffersize=1000)
                  },
                  loop = NULL,
                  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
                    if (length(unique(y)) <= 2)
                        out <- initOLK(d = dim(x)[2], kernel = "rbfdot", kpar = list(param$sigma),
                                       type = "classification", buffersize = param$buffersize)
                    else
                        out <- initOLK(d = dim(x)[2], kernel = "rbfdot", kpar = list(param$sigma),
                                       type = "regression", buffersize = param$buffersize)
                    out <- olk(out, x = as.matrix(x), y = y, C = param$C, r = param$r, epsilon = param$epsilon)
                    out
                    },
                  predict = function(modelFit, newdata, submodels = NULL) {
                    olkPred <- function(obj, x)
                    {
                      pred <- predict(obj, x)
                      pred <- factor((pred > 0)*1 + 1)
                      levels(pred) <- lev(obj)
                      pred
                    }
                    out <- try(olkPred(modelFit, newdata), silent = FALSE)
                    out
                  },
                  prob = function(modelFit, newdata, submodels = NULL) {
                    olkPred <- function(obj, x)
                    {
                      pred <- predict(obj, x)
                      pred
                    }
                    out <- try(olkPred(modelFit, newdata), silent = FALSE)
                    # scale between 0 and 1 with sigmoid, scaling power? use sigma?
                    out <- apply(as.matrix(out), MARGIN = 2, FUN = function(x) 1/(1 + exp(x)))
                    out <- data.frame(out)
                    names(out) <- lev(modelFit)[1]
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
                    # If the cost is high, the decision boundary will work hard to
                    # adapt. Also, if C is fixed, smaller values of sigma yeild more
                    # complex boundaries
                    x[order(x$C),]
                  })
