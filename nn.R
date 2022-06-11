num.grad <- function(f, x) {
  fx <- f(x)
  grad <- rep(0, length(x))
  h <- 0.00001
  for (i in 1:length(x)) {
    xi <- x[i]
    fxph <- f(xi + h)
    fxmh <- f(xi -h)
    grad[i] <- (fxph - fxmh) / (2 * h)
  }
  return(grad)
}

relu <- function(x) {
  return(pmax(x, 0))
}

softmax <- function(scores) {
  es <- exp(scores)
  return(es / rowSums(es))
}

argmax <- function(v) {
  i.max <- 1
  for (i in 1:length(v)) {
    if (v[i] > v[i.max]) {
      i.max <- i
    }
  }
  return(i.max)
}

generate.spiral <- function(N, D, K) {
  M <- N*K # Total number of examples
  X <- matrix(0, M,D) # data matrix (each row = single example)
  y <- rep(0, M) # class labels
  for (j in 1:K){
    r <- 0
    t <- j * 4
    start <-N*(j-1) +1
    end <- N*j
    for (k in start:end) {
      tp <- t+ runif(1) * 0.2
      X[k, 1] <- r * sin(tp)
      X[k, 2] <- r * cos(tp)
      y[k] <- j
      r <- r + 1/N
      t <- t + 4/N
    }
  }
  return(list(X = X, y = y))
}

parse.iris <- function() {
  X <- as.matrix(iris[,1:4])
  y <- as.integer(as.factor(iris[,5]))
  return(list(X = X, y = y))
}

#num <- 100 # number of points per class
#dim <- 2 # dimensionality
#class <- 3 # number of classes
#Xy <- generate.spiral(num, dim, class)
Xy <- parse.iris()
X <- Xy$X
y <- Xy$y
# visualize the data
#plot(X[, 1], X[, 2], col=y)

D <- ncol(X)
M <- nrow(X)
K <- max(y)
# Initialize weights and biases
# Neural network
H <- 128 # Hidden layer size
W1 <- matrix(rnorm(D*H), nrow=D) * 0.01
b1 <- rep(0, H)
W2 <- matrix(rnorm(H*K), nrow=H) * 0.01
b2 <- rep(0, K)
# Softmax classifier
#W <- matrix(rnorm(D*K), nrow= D) * 0.01
#b <- rep(0, K)
lambda <- 0.0 # Regularization constant
alpha <- 0.8 # Step size
batch.size <- M/10
maxit <- 100 * M / batch.size
loss.arr <- rep(0, maxit)

for (it in 1:maxit) {
  randperm <- sample(1:length(y))
  epoch.loss <- 0
  for (batch in 1:(M / batch.size)) {
    batch.begin <- batch.size * (batch - 1) + 1
    batch.end <- batch.size * batch
    Xbatch <- X[randperm, ][batch.begin:batch.end,]
    ybatch <- y[randperm][batch.begin:batch.end]
    # NN
    hidden.layer <- relu(Xbatch %*% W1 + b1)
    scores <-  hidden.layer %*% W2 + b2 # Pass through NN
    # Softmax
    # scores <- X %*% W + b
    probs <- softmax(scores) # Softmax so the probs sum to 1
    correct.logprob <- rep(0, batch.size)
    for (i in 1:batch.size) { # For each example
      correct.logprob[i] <- -log(probs[i, ybatch[i]]) # Take the log prob of the correct class
    }
    data.loss <- sum(correct.logprob) / M
    reg.loss <- 0.5 * lambda * sum(W * W)
    loss <- data.loss + reg.loss
    #cat("it = ", it, ", loss = ", loss, "\n")
    epoch.loss <- epoch.loss + loss
    
    dscores <- probs
    for (i in 1:batch.size) {
      dscores[i, ybatch[i]] <- dscores[i, ybatch[i]] - 1
    }
    dscores <- dscores / M
    # Softmax backprop
    #dW <- t(X) %*% dscores
    #db <- rowSums(dscores)
    #dW <- dW + W * lambda
    # NN backprop
    dW2 <- t(hidden.layer) %*% dscores
    db2 <- rowSums(dscores)
    dhidden <- dscores %*% t(W2)
    for (i in 1:nrow(dhidden)) {
      for (j in 1:ncol(dhidden)) {
        if (hidden.layer[i, j] <= 0) {
          dhidden[i, j] <- 0 # ReLU backprop
        }
      }
    }
    dW1 <- t(Xbatch) %*% dhidden
    db1 <- rowSums(dhidden)
  
    W1 <- W1 - alpha * dW1
    b1 <- b1 - alpha * db1
    W2 <- W2 - alpha * dW2
    b2 <- b2 - alpha * db2
  }
  loss <- epoch.loss / (M / batch.size)
  loss.arr[it] <- loss
  cat("Epoch ", it, ", loss = ", loss, "\n")
}

y.pred <- rep(0, M)
hidden.layer <- relu(X %*% W1 + b1)
scores <-  hidden.layer %*% W2 + b2 
for (i in 1:M) {
  y.pred[i] <- argmax(scores[i,])
}
cat("Accuracy = ", sum(y.pred == y) / length(y), "\n")

plot(loss.arr, type="l")

par(mfrow=c(1, 2))
plot(prcomp(X)$x[,1:2], col=y)
plot(prcomp(X)$x[,1:2], col=y.pred)
par(mfrow=c(1, 1))

