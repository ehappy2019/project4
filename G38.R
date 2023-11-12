
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
  
  # the matrix W[[l]] has dimension d[l+1]xd[l], 
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
  
  # the first layer of the network is given by the vector 'inp'
  h[[1]] <- inp
  
  # the other layers are calculated using the W matrices and b vectors so that
  # h[[l+1]][j] = max(0, W[[l]][j,]%*%h[[l]] + b[[l]][j]) 
  for (l in 1:(length(h)-1)){
    h[[l+1]] <- W[[l]] %*% h[[l]] + b[[l]]
    h[[l+1]][h[[l+1]] < 0] <- 0 # elements that are negative are replaced by 0
  }
  
  list(h=h, W=W, b=b)
}

backward <- function(nn,k) {
  # for ease of notation
  h <- nn$h ; W <- nn$W
  
  # initialise some elements
  dh <- db <- dW <- list()
  L <- length(h)
  
  # compute dh[[L]] where L is the last layer 
  exp_L <- exp(h[[L]])
  sum_L <- sum(exp_L)
  dh[[L]] <- exp_L/sum_L
  dh[[L]][k] <- dh[[L]][k] - 1 # subtract 1 when j = k
  
  # for the other layers, work from the last layer to the first one 
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
  
  # choosing which rows of 'inp' matrix to use
  n <- nrow(inp)
  indices <- sample(1:n, mb, replace=FALSE)
  
  # for each input row, go through forwards and backwards 'nstep' times
  for (i in indices){
    x <- inp[i,]
    for (t in 1:nstep){
      nn <- forward(nn,x)
      nn <- backward(nn,k[i])
      W <- nn$W ; b <- nn$b ; dW <- nn$dW ; db <- nn$db
      for (l in 1:(length(nn$h)-1)){
        W[[l]] <- W[[l]] - eta*dW[[l]]
        b[[l]] <- b[[l]] - eta*db[[l]]
      }
      nn$W <- W ; nn$b <- b
    }
  }
  nn
}


# load the 'iris' data set
data(iris)

# separate it into training and test data
iris_rows <- 1:nrow(iris)
test_indices <- iris_rows[iris_rows%%5 == 0] # multiples of 5
training_indices <- iris_rows[!(iris_rows%%5 == 0)] # other indices

test_iris <- iris[test_indices,]
training_iris <- as.matrix(iris[training_indices,-5])
row.names(training_iris)<- NULL ; colnames(training_iris)<- NULL

# create vector k
k <- match(iris$Species, c('setosa', 'versicolor', 'virginica'))
k <- k[training_indices]


# tests
d <- c(4,8,7,3)
nn <- netup(d)
test <- train(nn, training_iris, k)



