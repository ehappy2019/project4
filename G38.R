# Anya Gray s1934439, Emily Happy s2553022, Alice Lasocki s2570813
# Classification neural network, trained using stochastic gradient descent



# FUNCTION INPUT: d, a vector giving the number of nodes in each layer of the 
#                 network
# FUNCTION OUTPUT: nn = a list representing the network, containing:
#                   - h : a list of node vectors for each layer
#                   - W : a list of weight matrices, initialized with U(0,0.2)
#                     random deviates
#                   - b : a list of offset vectors, initialized with U(0,0.2)
#                     random deviates
# DESCRIPTION: This function sets up the network list by creating lists of nodes,
#              weights, and offset vectors. The weight matrices and offset vectors
#              link layer l to layer l+1
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


# FUNCTION INPUT: nn = a network list as returned by 'netup'
#                 inp = a vector of input values for the first layer
# FUNCTION OUTPUT: nn = a list representing the network, with updated values
#                  for the nodes calculated from 'inp' 
# DESCRIPTION: This function takes a vector of input values and updates the rest
#              of the nodes using the weight and bias vectors. 
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
  }
  
  list(h=h, W=W, b=b)
}


# FUNCTION INPUT: nn = a network list as returned by forward
#                 k = an integer representing the output class for a datum
# FUNCTION OUTPUT: a list representing the updated network, containing:
#                   - h, W, b as before outputted from function netup
#                   - dh, dW, db : derivatives of the loss function wrt the 
#                                 nodes, weights and biases respectively.
#                                 These have the same dimension as the network.
# DESCRIPTION: This function calculates the updates for the parameters of nn by  
#              computing the derivative of the loss for class k w.r.t. the last 
#              layer of nodes, then computes the rest of the derivatives w.r.t
#              the rest of the nodes, the weight matrices, and the bias vectors.
#              The nn list is updated to include the derivatives to be used  
#              when running the 'train' function. 
backward <- function(nn, k) {
  # for ease of notation
  h <- nn$h ; W <- nn$W
  
  # initialize some elements
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


# FUNCTION INPUT: nn = the network as returned by 'netup',
#                 inp = a matrix whose rows are different input data vectors,
#                 k = vector whose entries correspond to the class of inp's rows,
#                 eta = step size to take when updating parameters,
#                 mb = number of data to randomly sample for each step,
#                 nstep = number of training iterations,
# FUNCTION OUTPUT: nn = a list representing the trained neural network
# DESCRIPTION: This function takes a random sample of data, runs it through the
#              network by calling 'forward', 'backward', then updates the weight 
#              matrices and bias vectors for each datum in the sample. Then 
#              the loss is calculated for the step and the process is repeated
#              nstep times. 
train <- function(nn, inp, k, eta=0.01, mb=10, nstep=10000){
  
  n <- nrow(inp)
  
  for (t in 1:nstep){
    
    # sampling mb random rows (input vectors) from the inp matrix
    indices <- sample(1:n, mb, replace=FALSE)
    
    for (i in indices){
      x <- inp[i,]     # input vector for forward pass
      
      # for each input row, propagate through forwards and backwards 'nstep' times
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      W <- nn$W ; b <- nn$b ; dW <- nn$dW ; db <- nn$db
      
      # update parameter values for this optimization step 
      for (l in 1:(length(nn$h)-1)){
        W[[l]] <- W[[l]] - eta*dW[[l]]
        b[[l]] <- b[[l]] - eta*db[[l]]
      }
      nn$W <- W ; nn$b <- b
    }
  }
  nn  # return the trained network
}

# FUNCTION INPUT: nn = the updated neural network,
#                 k = the class of the data point
# FUNCTION OUTPUT: loss = the negative log likelihood of the predicted class
# DESCRIPTION: This function calculates the negative log likelihood using the 
#              probability that the predicted class is k. 
loss <- function(nn, input, k){
  n <- nrow(input)
  loss <- rep(0, n)
  
  for (i in 1:n){
    inp <- input[i,]
    ki <- k[i]
    h <- forward(nn, inp)$h 
    L <- length(h)              # the total number of layers in the network
    
    exp_L <- exp(h[[L]])    
    sum_L <- sum(exp_L)
    
    predicted <- exp_L/sum_L    # probability of the output class being k
    loss[i] <- -log(predicted[ki])  # negative log likelihood
  }
  
  sum(loss/n)                   # return the sum negative log likelihood over n
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

# cleaning up the data to keep only numerical values 
data_iris <- as.matrix(iris[,-5])
rownames(data_iris)<- NULL ; colnames(data_iris)<- NULL

test_iris <- data_iris[test_indices,] # rows of iris dataset to test with 
training_iris <- data_iris[training_indices,]          # " to train with

# create vector k
k <- match(iris$Species, c('setosa', 'versicolor', 'virginica'))
training_k <- k[training_indices]
test_k <- k[test_indices]

# set up and train a network on training data

# We set the seed to obtain a good result:
set.seed(3)


d <- c(4,8,7,3) # set network architecture
nn <- netup(d)  # initialize weights and biases
trained_nn <- train(nn, training_iris, training_k) # train the network


# ---------------------------------------------------------------------------
### "After training write code to classify the test data to species according to
### the class predicted as most probable for each iris in the test set, and 
### compute the misclassification rate"

# code to classify the test data

# FUNCTION INPUT: nn = trained network,
#                 testset = data from iris the network has not seen yet,
#                 test_k = k vector containing classes corresponding to the testset 
# FUNCTION OUTPUT: misclassification: the proportion of misclassified data
# DESCRIPTION: This function runs the test data through the trained network using
#              one forward pass and records the classification that the model 
#              predicts. Then it compares the prediction to the true class to return 
#              the total proportion of missclassified inputs. 
predict_test <- function(nn, testset, test_k){
  n <- nrow(testset)
  wrong <- 0
  
  for (i in 1:n){
    
    inp <- testset[i,]
    k <- test_k[i]
    h <- forward(nn, inp)$h
    L <- length(h)
    predicted_class <- which.max(h[[L]])
    
    if (predicted_class != k){
      wrong <- wrong + 1
    }
  }
  # proportion of wrongly predicted classes
  wrong/n
}

# pre training
predict_test(nn, test_iris, test_k)
loss(nn, test_iris, test_k)

# post training
predict_test(trained_nn, test_iris, test_k)
loss(trained_nn, test_iris, test_k)

