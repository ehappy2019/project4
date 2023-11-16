# Anya Gray s1934439, Emily Happy s2553022, Alice Lasocki s2570813
# https://github.com/ehappy2019/project4


# CONTRIBUTIONS:
# We worked collaboratively as a group on this assignment. Emily created the 
# forward function, and all 3 of us collaborated creating the backward function.
# Alice did finite differencing testing. The code for training and testing the 
# data was written mainly by Alice and Emily. All 3 of us debugged and 
# fine-tuned the code, and Anya finalised the comments and presentation. 
# Proportions were more or less equal.



# This program creates a neural network from scratch. It is trained using the 
# ReLU transform as the activation function, and the negative log-likelihood of 
# the output probabilities as the loss function. Stochastic gradient descent is 
# used to optimise the parameters. In this case, we use it to classify data from
# the 'iris' dataset into species categories.

# Each layer of the network has a number of nodes and linking each layer, there 
# are free parameters known as a weight 'W' (which is matrix) and a bias 'b'
# (which is a vector). These parameters are drawn from stochastic normal 
# distributions at initialization of the network; the function 'netup' sets up 
# the network by creating the neuron layers, weight matrices and bias vectors 
# with the right dimensions. 

# The function 'forward' simulates the propagation of input data through the 
# network: a linear transformation is performed using the values of each neuron 
# in layer l with the relevant weights and biases, the ReLU activation is then 
# calculated, which determines the values of the neurons in the following layer 
# l+1:  h[[l+1]] = max(0, W[[l]] %*% h[[l]] + b[[l]]). This works recursively 
# until we reach the final layer, where the values of the neurons indicate 
# probabilities that the input datum is each of the possible classes. 

# The function 'backward' exercises back-propagation to calculate the 
# derivatives of the loss w.r.t the layers of neurons (dh), the weights (dW) and 
# the biases (db) according to the following formulas:
# for a pair (x,k) and network returned by using h[[1]] = x:
# dh[[L]][j] = exp(h[[L]][j])/sum(exp(h[[L]][i])) - kronecker_delta(k,j)
#   where the sum is over i = 1:length(h[[L]]), L is the output layer
# for l < L :
# d[[l+1]][j] = dh[[l+1]][j] if h[[l+1]][j] > 0 *OR* = 0 if h[[l+1]][j] <= 0
# dh[[l]] = W[[l]]^T %*% d[[l+1]]
# db[[l]] = d[[l+1]]
# dW[[l]] = d[[l+1]] %*% h[[l]]^T

# We train the network in the function 'train' by doing this forward and 
# backward propagation nstep times for mb pieces of test data each time (nstep 
# and mb are given default values 10000 and 10 respectively) . This is the 
# process that optimises the network's predictions by fine-tuning the weights 
# and biases at each step by using the derivatives of the loss, in order to 
# minimise the loss function. 
# The weights and biases are updated at each step as so :
# W[[l]] <- W[[l]] - eta*dW[[l]]
# b[[l]] <- b[[l]] - eta*db[[l]]
# where eta is the stepsize and db, dW are the average derivatives for the mb 
# randomly sampled data inputs used in one step.

# The function 'loss' calculates the loss function for a given network (either
# trained or not trained), the function 'classify' returns a list of predicted
# classifications for a given test set of data, and 'misclassification' measures
# the proportion of misclassifications for a given test set.

# We will use these functions to compare the results we get when testing our 
# trained neural network and an untrained on 'iris' data.



# ---------------------------------------------------------------------------
###FUNCTIONS



# FUNCTION INPUT: d = vector giving the number of neurons in each layer of the 
#                 network
# FUNCTION OUTPUT: nn = list representing the network, containing:
#                   - h = list of vectors for each layer with neuron values 
#                   - W = list of weight matrices, initialized with U(0,0.2)
#                     random deviates
#                   - b = list of bias vectors, initialized with U(0,0.2)
#                     random deviates
# DESCRIPTION: This function initializes the network by creating lists of 
#              neurons, weights, and biases. The weight matrices and bias 
#              vectors link layer l to layer l+1.
netup <- function(d) {
  
  # initialising network data
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
# FUNCTION OUTPUT: nn = a list representing the network, with the same elements 
#                  as the input nn, but with updated values for every neuron in 
#                  the network 
# DESCRIPTION: This function takes a vector of input values and updates the rest
#              of the neurons using the weight and bias vectors and the ReLU
#              activation function defined as h^{l+1} = max(0, W^l * h^l + b^l),
#              where h^l is the vector of neuron valuesin layer l. 
forward <- function(nn, inp) {
  
  # for ease of notation
  h <- nn$h ; W <- nn$W ; b <- nn$b
  
  # the first layer of the network is given by the input vector 'inp'
  h[[1]] <- inp
  
  # propagate the input through the rest of the network with the ReLU function
  for (l in 1:(length(h)-1)){
    h[[l+1]] <- W[[l]] %*% h[[l]] + b[[l]]
    h[[l+1]][h[[l+1]] < 0] <- 0 # elements that are negative are replaced by 0
  }
  
  # return updated network after one forward pass
  list(h=h, W=W, b=b)
}


# FUNCTION INPUT: nn = a network list as returned by forward
#                 k = an integer representing the output class of a datum
# FUNCTION OUTPUT: a list representing the updated network, containing:
#                   - h, W, b as before outputted from function netup
#                   - dh, dW, db : derivatives of the loss function wrt the 
#                                 nodes, weights and biases respectively.
#                                 These have the same dimension as the network
#                                 architecture (same dimensions as h).
# DESCRIPTION: This function calculates the parameter updates of nn by  
#              computing the derivative of the loss for class k. This works 
#              backwards by calculating the derivative of the loss  w.r.t. the 
#              last layer of nodes, then for each layer preceding the last, the 
#              derivative is computed by combining derivatives w.r.t the node
#              values, the weight matrices, and the bias vectors. The nn list is 
#              updated to include the derivatives to be used when running the 
#              'train' function. 
backward <- function(nn, k) {
  # for ease of notation
  h <- nn$h ; W <- nn$W
  
  # initialize derivatives
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
    d[h[[l+1]] <= 0] <- 0  # set d = 0 where h <= 0
    dh[[l]] <- t(W[[l]])%*%d
    
    db[[l]] <- d
    dW[[l]] <- d%*%t(h[[l]])
  }
  
  # return relevant parameters of the network and derivatives
  list(h=h, W=W, b=nn$b, dh=dh, dW=dW, db=db)
}


# FUNCTION INPUT: nn = the network as returned by 'netup'
#                 inp = a matrix whose rows are input data vectors
#                 k = vector whose entries correspond to the class of inp's rows
#                 eta = step size to take when updating parameters
#                 mb = number of test data to randomly sample for each step
#                 nstep = number of training iterations
# FUNCTION OUTPUT: nn = a list representing the trained neural network
# DESCRIPTION: This function takes a random sample of data, runs each datum 
#              through the network by calling 'forward' then 'backward'. It then 
#              updates the weight matrices and bias vectors for each datum in 
#              the sample, using the optimised values as returned by 'backward'. 
#              The process is repeated nstep times, to obtain an optimised 
#              network trained generally. 
train <- function(nn, inp, k, eta=0.01, mb=10, nstep=10000){
  
  n <- nrow(inp) # number of data
  L <- length(nn$h) # number of layers in the network
  
  # train the network for nstep iterations
  for (t in 1:nstep){
    
    # sampling mb random rows (input vectors) from the inp matrix
    indices <- sample(1:n, mb, replace=FALSE)
    
    # new lists will be updated to hold the optimized parameters until we 
    # replace W and b in the network
    new_W <- nn$W ; new_b <- nn$b
    
    # optimize parameters for each data of the random sample
    for (i in indices){
      x <- inp[i,]     # input vector for forward pass
      
      # for each input row, propagate through forwards and backwards 
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      dW <- nn$dW ; db <- nn$db
      
      # update new parameter values for this optimization step, dividing by mb 
      # so that once we go through all mb data it will be like having updated W 
      # and b by minus eta times the average of the dW's and db's respectively
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


# FUNCTION INPUT: nn = the neural network of which to calculate the loss,
#                 input = data used to obtain an output layer from which to 
#                         calculate the loss,
#                 k = the class of the data in 'input'
# FUNCTION OUTPUT: the total loss for given 'nn' and 'input' data
# DESCRIPTION: This function calculates the value of the loss function for the 
#              given network as a negative log-likelihood. It is used towards 
#              the end of the program to compare our pre-trained and 
#              post-trained networks.
loss <- function(nn, input, k){
  
  n <- nrow(input) # number of input data
  L <- length(nn$h) # the total number of layers in the network
  loss <- rep(0, n) # initialize vector to hold the loss for each row of 'input'
  
  # for each input data
  for (i in 1:n){
    
    # run the data through forward to obtain corresponding output layer
    inp <- input[i,]
    ki <- k[i]
    output <- forward(nn, inp)$h 
    
    # calculate probability that the output variable is in class k : 'predicted'
    exp_L <- exp(output[[L]])    
    sum_L <- sum(exp_L)
    predicted <- exp_L/sum_L    
    
    # the loss for datum i is the negative log of predicted[ki] where ki 
    # is the true class
    loss[i] <- -log(predicted[ki])  
  }
  # the total loss is the mean of all the losses for each datum
  sum(loss/n)    
}



# ---------------------------------------------------------------------------
### Here we train a network to classify irises to species based on the 4 
### characteristics given in the iris data set.

# The iris network has 4 layers with the numbers of neurons as 4-8-7-3. The 
# input layer has 4 neurons corresponding to pieces of information about a 
# single datum, and the output layer has 3 neurons, corresponding to the 3 
# possible  species that the input datum may correspond to.


# load the 'iris' data set
data(iris)

# clean up the data to keep only numerical values 
data_iris <- as.matrix(iris[,-5])
rownames(data_iris)<- NULL ; colnames(data_iris)<- NULL

# separate it into training and test data
iris_rows <- 1:nrow(iris)
test_indices <- iris_rows[iris_rows%%5 == 0] # indices of multiples of 5
training_indices <- iris_rows[!(iris_rows%%5 == 0)] # all other indices

test_iris <- data_iris[test_indices,] # rows of iris dataset to test with 
training_iris <- data_iris[training_indices,]          # " to train with

# create vector k holding the classification of each iris datum
k <- match(iris$Species, c('setosa', 'versicolor', 'virginica'))

# splitting up the classifications for testing the network and training it
test_k <- k[test_indices]
training_k <- k[training_indices]


## Set up and train a network using training data

# set the seed for the stochastic variables (weights and biases) 
# to obtain a good result:
set.seed(3)

d <- c(4,8,7,3) # set up the network architecture
nn <- netup(d)  # initialize weights and biases
trained_nn <- train(nn, training_iris, training_k) # train the network

# ---------------------------------------------------------------------------
### Now that we have trained the network, we will use it to classify the species
### of the test data. The species predictions are taken as the highest node 
### value in the output layer, after the test data is propagated through the 
### trained network. We then calculate the misclassification rate of our 
### network.



# FUNCTION INPUT: nn = neural network,
#                 input = data to predict a class for using the network, each 
#                 row is a data input
# FUNCTION OUTPUT: predicted_k = vector of predicted class for each row of input
# DESCRIPTION: This function uses the neural network 'nn' and runs each input
#              datum through the 'forward' function with network 'nn' to 
#              predict the class of each datum. It returns the predicted classes 
#              of all the input data in a vector.
classify <- function(nn, input){
  
  n <- nrow(input) # number of input data
  L <- length(nn$h) # total number of layers in the network
  predicted_k <- rep(0, n) # initialize vector to store predicted class of data
  
  # for each data set, run it through forward and get predicted class based on 
  # which node of the output layer has the highest value
  for (i in 1:n){
    inp <- input[i,]
    output <- forward(nn, inp)$h
    predicted_k[i] <- which.max(output[[L]])
  }
  
  # return the list of predicted classes
  predicted_k
}


# FUNCTION INPUT: predicted_k = vector of classes predicted by a neural network
#                 actual_k = true classes of the input data used to obtain 
#                 'predicted_k'
# FUNCTION OUTPUT: the misclassification rate (a number between 0 and 1)
# DESCRIPTION: This function calculates the misclassification rate of our 
#              trained neural network. It compares the classes given in 
#              'predicted_k' and 'actual_k' and returns the proportion of
#              wrongly predicted classes.
misclassified <- function(predicted_k, actual_k){
  length(which(predicted_k != actual_k))/length(actual_k)
}


# ---------------------------------------------------------------------------
###RESULTS

# pre training results : misclassification rate and loss
pre_classify <- misclassified(classify(nn, test_iris), test_k)
pre_loss <- loss(nn, test_iris, test_k)
cat('Before training the network, there was a', pre_classify,
    'misclassification rate and a loss of', pre_loss)

# post training results : misclassification rate and loss
post_classify <- misclassified(classify(trained_nn, test_iris), test_k)
post_loss <- loss(trained_nn, test_iris, test_k)
cat('\n\nAfter training the network, there is now a misclassification rate of',
    post_classify, 'and a loss of', post_loss)

