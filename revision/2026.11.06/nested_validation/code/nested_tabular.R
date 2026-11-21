# All predictions are computed without passing heldout outcomes into this process.
suppressPackageStartupMessages({library(glmnet); library(jsonlite); library(caret)})
Sys.setenv(OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, TZ="Asia/Shanghai")
glmnet.control(fdev=0, devmax=1)

write_json <- function(x,p) {
  tmp <- paste0(p,".tmp")
  jsonlite::write_json(x,tmp,auto_unbox=TRUE,pretty=TRUE,null="null",na="null",digits=16)
  if (!file.rename(tmp,p)) stop("Atomic JSON rename failed")
}
auc <- function(y,p) {
  n1 <- sum(y==1);n0 <- sum(y==0)
  if (n1==0 || n0==0 || any(!is.finite(p))) stop("Invalid AUC inputs")
  (sum(rank(p,ties.method="average")[y==1])-n1*(n1+1)/2)/(n1*n0)
}
fold_id <- function(y,k,seed) {
  stopifnot(min(table(y))>=k);set.seed(seed);f <- integer(length(y))
  for (v in c(0,1)) {ix <- which(y==v);f[ix] <- sample(rep(seq_len(k),length.out=length(ix)))}
  f
}
prep_fit <- function(x,cor_filter) {
  x <- as.data.frame(x,check.names=FALSE)
  keep <- names(x)[vapply(x,function(z) mean(is.na(z))<=.20,logical(1))]
  x <- x[,keep,drop=FALSE]
  med <- vapply(x,median,numeric(1),na.rm=TRUE)
  for (n in names(x)) x[[n]][is.na(x[[n]])] <- med[[n]]
  v <- vapply(x,var,numeric(1));keep <- names(v)[is.finite(v)&v>0]
  x <- x[,keep,drop=FALSE]
  if (cor_filter && ncol(x)>1) {
    cc <- cor(x);cc[!is.finite(cc)] <- 0;diag(cc) <- 1
    drop <- caret::findCorrelation(cc,cutoff=.95,names=TRUE,exact=TRUE)
    x <- x[,setdiff(names(x),drop),drop=FALSE]
  }
  mu <- vapply(x,mean,numeric(1));sdv <- vapply(x,sd,numeric(1))
  list(features=names(x),median=med[names(x)],mean=mu,sd=sdv)
}
prep_apply <- function(x,prep) {
  x <- as.data.frame(x[,prep$features,drop=FALSE],check.names=FALSE)
  for (n in names(x)) x[[n]][is.na(x[[n]])] <- prep$median[[n]]
  z <- as.matrix(x)
  if (ncol(z)>0) z <- sweep(sweep(z,2,prep$mean,"-"),2,prep$sd,"/")
  # glmnet requires at least two columns; zero padding carries no information.
  while (ncol(z)<2) {z <- cbind(z,rep(0,nrow(z)));colnames(z)[ncol(z)] <- paste0(".zero_padding_",ncol(z))}
  z
}
path_fit <- function(x,y,alpha,lambdas) {
  if (all(apply(x,2,var)==0)) return(list(constant=mean(y),lambda=lambdas))
  fit <- glmnet(x,y,family="binomial",alpha=alpha,lambda=lambdas,
                standardize=FALSE,intercept=TRUE,thresh=1e-7,maxit=100000)
  if (fit$jerr!=0 || min(fit$lambda)>min(lambdas)*(1+1e-6)) stop("Incomplete glmnet lambda path; refusing silent extrapolation")
  list(model=fit,lambda=lambdas)
}
path_predict <- function(fit,x,lambdas,type="response") {
  if (!is.null(fit$constant)) {
    p <- if (type=="link") qlogis(fit$constant) else fit$constant
    return(matrix(p,nrow(x),length(lambdas)))
  }
  as.matrix(predict(fit$model,newx=x,s=lambdas,type=type))
}
tune_fit <- function(x,y,ids,alphas,lambdas,k,seed,cor_filter,directory) {
  dir.create(directory,recursive=TRUE,showWarnings=FALSE)
  f <- fold_id(y,k,seed)
  write.csv(data.frame(ID=ids,inner_fold=f),file.path(directory,"tuning_assignments.csv"),row.names=FALSE)
  scores <- array(NA_real_,c(length(alphas),length(lambdas),k))
  retained <- list()
  for (j in seq_len(k)) {
    tr <- which(f!=j);va <- which(f==j)
    prep <- prep_fit(x[tr,,drop=FALSE],cor_filter)
    xt <- prep_apply(x[tr,,drop=FALSE],prep); xv <- prep_apply(x[va,,drop=FALSE],prep)
    retained[[j]] <- list(fold=j,fit_ids=as.list(ids[tr]),validation_ids=as.list(ids[va]),features=as.list(prep$features))
    for (a in seq_along(alphas)) {
      fit <- path_fit(xt,y[tr],alphas[a],lambdas)
      pp <- path_predict(fit,xv,lambdas)
      scores[a,,j] <- vapply(seq_along(lambdas),function(l) auc(y[va],pp[,l]),numeric(1))
    }
  }
  grid <- list();selected <- list()
  for (a in seq_along(alphas)) {
    values <- matrix(scores[a,,],nrow=length(lambdas),ncol=k)
    means <- rowMeans(values);ses <- apply(values,1,sd)/sqrt(k)
    peak <- which.max(means)
    chosen <- which(means>=means[peak]-ses[peak])[1] # lambda grid is descending
    grid[[a]] <- data.frame(alpha=alphas[a],lambda=lambdas,mean_auc=means,se_auc=ses,
                            lambda_1se_selected=seq_along(lambdas)==chosen)
    selected[[a]] <- data.frame(alpha=alphas[a],lambda=lambdas[chosen],peak_auc=means[peak],
                               selected_auc=means[chosen],selected_se=ses[chosen])
  }
  candidates <- do.call(rbind,selected)
  best <- candidates[order(-candidates$peak_auc,-candidates$selected_auc,-candidates$alpha)[1],]
  write.csv(do.call(rbind,grid),file.path(directory,"tuning_grid.csv"),row.names=FALSE)
  write_json(retained,file.path(directory,"preprocessing_boundaries.json"))
  prep <- prep_fit(x,cor_filter);xt <- prep_apply(x,prep)
  fit <- path_fit(xt,y,best$alpha,lambdas)
  if (is.null(fit$constant)) {
    beta <- as.matrix(coef(fit$model,s=best$lambda))
    names_beta <- rownames(beta)
    keep <- names_beta[as.numeric(beta)!=0 & names_beta!="(Intercept)" & !startsWith(names_beta,".zero_padding_")]
    coefficients <- data.frame(feature=names_beta,coefficient=as.numeric(beta))
  } else {keep <- character(0);coefficients <- data.frame(feature="(Intercept)",coefficient=qlogis(fit$constant))}
  write.csv(coefficients,file.path(directory,"coefficients.csv"),row.names=FALSE)
  selection <- list(alpha=best$alpha,lambda=best$lambda,peak_inner_auc=best$peak_auc,
                    selected_inner_auc=best$selected_auc,selected_features=as.list(keep),
                    preprocessing_features=as.list(prep$features),training_ids=as.list(ids),
                    selected_at_grid_boundary=best$lambda %in% range(lambdas))
  write_json(selection,file.path(directory,"selection.json"))
  result <- list(prep=prep,fit=fit,lambda=best$lambda,alpha=best$alpha,selected_features=keep,training_ids=ids)
  saveRDS(result,file.path(directory,"model.rds"))
  result
}
score_model <- function(model,x,type="link") {
  as.numeric(path_predict(model$fit,prep_apply(x,model$prep),model$lambda,type))
}

run <- function(directory,protocol_path,final=FALSE) {
  began <- proc.time()[3]
  protocol <- jsonlite::fromJSON(protocol_path,simplifyVector=TRUE)
  data <- read.csv(file.path(directory,"training.csv"),check.names=FALSE)
  pred <- read.csv(file.path(directory,"prediction_features.csv"),check.names=FALSE)
  stopifnot(!"event" %in% names(pred),!anyDuplicated(data$ID),!anyDuplicated(pred$ID))
  if (!final) stopifnot(!any(data$ID %in% pred$ID)) # cohort-specific IDs can overlap in final external inference
  split <- jsonlite::fromJSON(file.path(directory,"split.json"),simplifyVector=FALSE)
  seed <- if (final) 8900 else 8200+as.integer(split$fold)
  y <- data$event;stopifnot(all(y %in% c(0,1)))
  rad <- grep('^rad__',names(data),value=TRUE);cmr <- grep('^cmr__',names(data),value=TRUE)
  stopifnot(length(rad)==length(protocol$rscore_features),length(cmr)==length(protocol$cmr_candidates))
  model <- tune_fit(data[,rad,drop=FALSE],y,data$ID,protocol$alpha_grid,protocol$lambda_grid,
                    protocol$tuning_folds,seed,TRUE,file.path(directory,"rscore_full"))
  output <- data.frame(ID=pred$ID,Rscore=score_model(model,pred[,rad,drop=FALSE]),
                       Rscore_probability=score_model(model,pred[,rad,drop=FALSE],"response"))
  write.csv(output,file.path(directory,"rscore_prediction.csv"),row.names=FALSE)
  if (!final) {
    oof <- rep(NA_real_,nrow(data))
    for (s in split$rscore_crossfit) {
      ti <- match(unlist(s$training_ids),data$ID); vi <- match(unlist(s$heldout_ids),data$ID)
      stopifnot(!anyNA(ti),!anyNA(vi),length(intersect(ti,vi))==0,length(union(ti,vi))==nrow(data))
      child <- tune_fit(data[ti,rad,drop=FALSE],y[ti],data$ID[ti],protocol$alpha_grid,
                        protocol$lambda_grid,protocol$tuning_folds,seed+100*as.integer(s$split),TRUE,
                        file.path(directory,paste0("rscore_crossfit_",s$split)))
      stopifnot(all(is.na(oof[vi])))
      oof[vi] <- score_model(child,data[vi,rad,drop=FALSE])
      cat("Completed Rscore crossfit",s$split,"of 5 in",directory,"\n");flush.console()
    }
    stopifnot(all(is.finite(oof)))
    write.csv(data.frame(ID=data$ID,Rscore=oof),file.path(directory,"rscore_training_crossfit.csv"),row.names=FALSE)
  }
  cmr_model <- tune_fit(data[,cmr,drop=FALSE],y,data$ID,1,protocol$lambda_grid,
                       protocol$tuning_folds,seed+900, FALSE,file.path(directory,"cmr_selection"))
  stopifnot(!any(grepl('event|Rscore',cmr_model$selected_features)))
  write_json(list(completed=TRUE,seconds=unname(proc.time()[3]-began),training_n=nrow(data),
                  prediction_n=nrow(pred),fit_ids=as.list(data$ID),prediction_ids=as.list(pred$ID),
                  cmr_selected=as.list(cmr_model$selected_features),
                  R_version=R.version.string,glmnet_version=as.character(packageVersion('glmnet'))),
             file.path(directory,"r_complete.json"))
  capture.output(sessionInfo(),file=file.path(directory,"R_sessionInfo.txt"))
}

if (sys.nframe()==0) {
  args <- commandArgs(trailingOnly=TRUE)
  if (length(args)<2) stop("Usage: nested_tabular.R RUN_DIRECTORY PROTOCOL_JSON [final]")
  run(args[1],args[2],length(args)>2 && args[3]=="final")
}
