# train with averaged derivatives

train <- function(nn, inp, k, eta=0.01, mb=10, nstep=100){
  
  n <- nrow(inp) # number of data
  L <- length(nn$h) # number of layers in the network
  
  # lists to hold the derivatives for each data point
  dW_list <- list(); average_dW <- list()
  db_list <- list(); average_db <- list()
  
  for (l in 1:(L-1)){
    average_dW[[l]] <- matrix(0, dim(nn$W[[l]])[1], dim(nn$W[[l]])[2])
    average_db[[l]] <- rep(0, length(nn$b[[l]]))
  }
  
  # train the network for nstep iterations
  for (t in 1:nstep){
    
    # sampling mb random rows (input vectors) from the inp matrix
    indices <- sample(1:n, mb, replace=FALSE)
    
    j = 1
    # optimize parameters for each of the random samples
    for (i in indices){
      x <- inp[i,]     # input vector for forward pass
      
      # for each input row, propagate through forwards and backwards 
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      W <- nn$W ; b <- nn$b ; dW_list[[j]] <- nn$dW ; db_list[[j]] <- nn$db
      j = j + 1
    }
    
    # find the average of each derivative
    for (l in 1:(L-1)){
      for (i in 1:mb){
        average_dW[[l]] = average_dW[[l]] + dW_list[[i]][[l]]/mb
        average_db[[l]] = average_db[[l]] + db_list[[i]][[l]]/mb
      }
    }
    
    # update parameter values with the average for this optimization step 
    for (l in 1:(L-1)){
      W[[l]] <- W[[l]] - eta*average_dW[[l]]
      b[[l]] <- b[[l]] - eta*average_db[[l]]
      nn$W <- W ; nn$b <- b
    }
    
  }
  
  # return the trained network
  nn 
}