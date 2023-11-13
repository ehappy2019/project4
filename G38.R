
# FUNCTION INPUT: d, a vector giving the number of nodes in each layer of the 
#                 network
# FUNCTION OUTPUT: a named list nn representing the network, containing:
#                   - h : a list of nodes for each layer
#                   - W : a list of weight matrices, initialized with U(0,0.2)
#                     random deviates
#                   - b : a list of offset vectors, initialized with U(0,0.2)
#                     random deviates
netup <- function(d) {
  h <- list()
  W <- list()
  b <- list()
  
  # each layer of the network contains d[l] nodes
  for (l in 1:length(d)){
    h[[l]] <- rep(1, d[l])
  }
  
  # the matrix W[[l]] has dimension (d[l+1] x d[l]), 
  # the vector b[[l]] has length d[l+1]
  for (l in 1:(length(d)-1)){
    W[[l]] <- matrix(runif(d[l+1]*d[l], min=0, max=0.2), nrow=d[l+1], ncol=d[l])
    b[[l]] <- runif(d[l+1], min=0, max=0.2)
  }
  
  list(h=h, W=W, b=b)
}


# FUNCTION INPUT: nn, a network list as returned by nn
#                 inp, a vector of input values for the first layer
# FUNCTION OUTPUT: a named list nn representing the network, with updated values
#                  for the nodes calculated from 'inp' 
forward <- function(nn, inp) {
  # for ease of notation
  h <- nn$h ; W <- nn$W ; b <- nn$b
  
  # the first layer of the network is given by the input vector 'inp'
  h[[1]] <- inp
  
  # the other layers are calculated using the W matrices and b vectors so that
  # h[[l+1]][j] = max(0, W[[l]][j,]%*%h[[l]] + b[[l]][j]) 
  for (l in 1:(length(h)-1)){
    h[[l+1]] <- W[[l]] %*% h[[l]] + b[[l]]
    h[[l+1]][h[[l+1]] < 0] <- 0 # elements that are negative are replaced by 0
    
    # perhaps more efficient way of doing the same thing? not sure
    # node <- W[[l]] %*% h[[l]] + b[[l]]
    # if (node > 0){
    #   h[[l+1]] <- node
    # } else { h[[l+1]] <- 0}
    
    ## alice : h[[l+1]] is actually the vector with all the nodes in layer l+1
    # in the previous version of the code we had it in a loop but I think doing 
    # it with vector operations like this is faster?
  }
  
  list(h=h, W=W, b=b)
}


# FUNCTION INPUT: nn, a network list as returned by nn
#                 k, an integer representing the output class for nn
# FUNCTION OUTPUT: a list representing the updated network, containing:
#                   - h, W, b as before outputted from function netup
#                   - dh, dW, db : derivatives of the loss function wrt the 
#                                 nodes, weights and biases respectively.
#                                 These have the same dimension as the network.
backward <- function(nn, k) {
  # for ease of notation
  h <- nn$h ; W <- nn$W
  
  # initialise some elements
  dh <- db <- dW <- list()
  
  # total number of layers in the network
  L <- length(h)
  
  # compute dh for the last layer: stored in dh[[L]]
  exp_L <- exp(h[[L]])
  sum_L <- sum(exp_L)
  dh[[L]] <- exp_L/sum_L
  dh[[L]][k] <- dh[[L]][k] - 1  # subtract 1 from node j when j = k
  
  ## BACK PROPAGATION
  # for layers l < L, work backwards from the last layer to the first
  for (l in (L-1):1){
    d <- dh[[l+1]] # d is d_(l+1)
    d[h[[l+1]] <= 0] <- 0  # set d=0 where h is <=0
    dh[[l]] <- t(W[[l]])%*%d
    
    db[[l]] <- d
    dW[[l]] <- d%*%t(h[[l]])
  }
  
  list(h=h, W=W, b=nn$b, dh=dh, dW=dW, db=db)
}


# FUNCTION DESCRIPTION: for each piece of input data i, we propagate i through
# the network by calling the function 'forward', and then train the network by 
# calling the function 'backward'. Then we update the weight and bias parameters
# and repeat nstep=10000 times.

# FUNCTION INPUT: nn, the network as returned by netup
#                 inp, a matrix whose rows are different input data vectors
#                 k, vector whose entries correspond to the class of inp's rows
#                 eta, step size to take when updating parameters
#                 mb = number of data to randomly sample for use in training
#                 nstep = number of training iterations per datum
# FUNCTION OUTPUT: 
train <- function(nn, inp, k, eta=0.01, mb=10, nstep=10000){
  
  # sampling mb random rows (input vectors) from the inp matrix
  n <- nrow(inp)
  indices <- sample(1:n, mb, replace=FALSE)
  
  # for each input row, propagate through forwards and backwards 'nstep' times
  for (i in indices){
    x <- inp[i,]
    for (t in 1:nstep){
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      W <- nn$W ; b <- nn$b ; dW <- nn$dW ; db <- nn$db
      
      # update parameter values for this timestep 
      # anya: why do we have length(h-1) here? 
      # alice: because there are only length(h)-1 matrices/vectors W and b
      # because they are 'between' each layer kinnd of so there's one less
      for (l in 1:(length(nn$h)-1)){
        W[[l]] <- W[[l]] - eta*dW[[l]]
        b[[l]] <- b[[l]] - eta*db[[l]]
      }
      nn$W <- W ; nn$b <- b
    }
  }
  nn
}


# ---------------------------------------------------------------------------
### "Train a 4-8-7-3 network to classify irises to species based on the 4
### characteristics given in the iris data set"

# load the 'iris' data set
data(iris)

# separate it into training and test data
iris_rows <- 1:nrow(iris)
test_indices <- iris_rows[iris_rows%%5 == 0] # multiples of 5
training_indices <- iris_rows[!(iris_rows%%5 == 0)] # other indices

test_iris <- as.matrix(iris[test_indices,-5]) # rows of iris dataset to test with 
training_iris <- as.matrix(iris[training_indices,-5])          # " to train with

# removing the row and column names from the dataset so we've just got the 
# numerical data to work with
rownames(training_iris)<- NULL ; colnames(training_iris)<- NULL
rownames(test_iris)<- NULL ; colnames(test_iris)<- NULL

# create vector k
# anya: i have put the indexing here on the first line as using the second line gave me some NAs in k? 
training_k <- match(iris$Species, c('setosa', 'versicolor', 'virginica'))[training_indices]
# k <- k[training_indices]
test_k <- match(iris$Species, c('setosa', 'versicolor', 'virginica'))[test_indices]

# tests
d <- c(4,8,7,3) # set network architecture
nn <- netup(d)  # initialize weights and biases
trained_nn <- train(nn, training_iris, training_k) # train the network


# ---------------------------------------------------------------------------
### "After training write code to classify the test data to species according to
### the class predicted as most probable for each iris in the test set, and 
### compute the misclassification rate"

# code to classify the test data

# DESCRIPTION: run the test data through the network and record which 
#              classification the model predicts. compare this to the true class
# INPUT: nn = trained network
#        testset = data from iris the network has not seen yet
#        test_k = k vector containing classes corresponding to the testset 

test <- function(nn, testset, test_k){
  n <- nrow(testset)
  correct <- 0
  for (i in 1:n){
    inp <- testset[i,]
    k <- test_k[i]
    result <- forward(nn, inp)
    h <- result$h
    L <- length(h)
    predicted_class <- which.max(h[[L]])
    if (predicted_class == k){
      correct <- correct + 1
    }
  }
  missclassification <- 1 - correct/n
  
  missclassification
}

# pre training
test(nn, test_iris, test_k)

# post training
test(trained_nn, test_iris, test_k)

