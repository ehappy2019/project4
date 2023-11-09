
netup <- function(d) {
  h <- list(0)
  W <- list(0)
  b <- list(0)
  
  for (l in 1:length(d)){
    h[[l]] <- c(rep(1, d[l]))
  }
  for (l in 1:(length(d)-1)){
    W[[l]] <- matrix(runif(d[l+1]*d[l], min=0, max=.2), nrow=d[l+1], ncol=d[l])
    b[[l]] <- runif(d[l+1], min=0, max=0.2)
  }
  list(h=h, W=W, b=b)
}

forward <- function(nn, inp) {
  for (i in 1:length(nn$h[[1]])){
    nn$h[[1]][i] <- inp[i]
  }
  for (l in 1:(length(nn$h)-1)){
    for (j in 1:length(nn$h[[l+1]])){
      value <- (nn$W[[l]][j] * nn$h[[l]]) + nn$b[[l]][j]
      nn$h[[l+1]][j] <- max(0, value)
    }
  }
  list(h=nn$h, W=nn$W, b=nn$b)
}

backward <- function(nn,k) {
  h <- nn$h ; W <- nn$W
  
  dh <- db <- dW <- list(0)
  L <- length(h)
  
  exp_L <- exp(h[[L]])
  sum_L <- sum(exp_L)
  dh[[L]] <- exp_L/sum_L
  dh[[L]][k] <- dh[[L]][k] - 1 
  
  for (l in (L-1):1){
    d <- dh[[l+1]]
    d[h[[l+1]] <= 0] <- 0 # d is d_(l+1)
    dh[[l]] <- t(W[[l]])%*%d
    
    db[[l]] <- d
    dW[[l]] <- d%*%t(h[[l]])
  }
  
  list(h=h, W=W, b=nn$b, dh=dh, dW=dW, db=db)
}

train <- function(nn, inp, k, eta=0.01, mb=10, nstep=10000){
  n <- nrow(inp)
  indices <- sample(1:n, mb, replace=FALSE)
  for (i in indices){
    x <- inp[i,]
    for (t in 1:nstep){
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      W <- nn$W ; b <- nn$b ; dW <- nn$dW ; db <- nn$db
      for (l in 1:length(nn$h)){
        W[[l]] <- W[[l]] - eta*dW[[l]]
        b[[l]] <- b[[l]] - eta*db[[l]]
      }
      nn$W <- W ; nn$b <- b
    }
  }
}


d <- c(4,8,7,3)
nn = backward(forward(netup(d), c(1, 2, 3, 4)))



