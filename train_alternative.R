train <- function(nn, inp, k, eta=0.01, mb=10, nstep=10000){
  
  n <- nrow(inp) # number of data
  L <- length(nn$h) # number of layers in the network
  
  # train the network for nstep iterations
  for (t in 1:nstep){
    
    # sampling mb random rows (input vectors) from the inp matrix
    indices <- sample(1:n, mb, replace=FALSE)
    
    # new vectors will be updated to hold the optimized parameters
    new_W <- nn$W ; new_b <- nn$b
    
    # optimize parameters for each data of the random sample
    for (i in indices){
      x <- inp[i,]     # input vector for forward pass
      
      # for each input row, propagate through forwards and backwards 
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      dW <- nn$dW ; db <- nn$db
      
      # update parameter values for this optimization step, dividing by mb so 
      # that once we go through all mb data it will be like having updated W by
      # the average of the dW's and db's
      for (l in 1:(L-1)){
        new_W[[l]] <- new_W[[l]] - eta*dW[[l]]/mb
        new_b[[l]] <- new_b[[l]] - eta*db[[l]]/mb
      }
    }
    # update W and b in the network
    nn$W <- new_W ; nn$b <- new_b
  }
  
  # return the trained network
  nn 
}