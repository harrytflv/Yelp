# Referrence Albert Au Yeung (2010)
# An implementation of matrix factorization

#' INPUT:
#' R     : a matrix to be factorized, dimension N x M
#' P     : an initial matrix of dimension N x K
#' Q     : an initial matrix of dimension M x K
#' K     : the number of latent features
#' steps : the maximum number of steps to perform the optimisation
#' alpha : the learning rate
#' beta  : the regularization parameter
#' OUTPUT:
#' the final matrices P and Q

matrix_factorization <- function (R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02){
  Q = t(Q)
  for(step in 1:steps){
    for(i in 1:length(R[,1])){
      for(j in 1:length(R[i,])){
        if(R[i,j] > 0){
          eij = R[i,j] - P[i,] %*% Q[,j]
          for(k in 1:K){
            P[i,k] = P[i,k] + alpha * (2 * eij * Q[k,j] - beta * P[i,k])
            Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])
          }
        }
      }
    }
    eR = P %*% Q
    e = 0
    for(i in 1:length(R[,1])){
      for(j in 1:length(R[i,])){
        if(R[i,j] > 0){
          e = e + (R[i,j] - (P[i,] %*% Q[,j]))^2
          for(k in 1:K){
            e = e + (beta/2) * ( (P[i,k])^2 + (Q[k,j])^2 )
          }
        }
      }
    }
    if(e < 0.001){
      break
    }
  }
  return(list(P, t(Q)))
}


###############################################################################

# Example
R <- matrix(c(5,3,0,1,4,0,0,1,1,1,0,5,1,0,0,4,0,1,5,4), nrow = 4, ncol = 5)
R <- t(R)
N = length(R[,1])
M = length(R[1,])
K = 2
P = matrix(rexp(N*K, rate=0.5), ncol = K)
Q = matrix(rexp(M*K, rate=0.5), ncol = K)

matrix_list <- matrix_factorization(R, P, Q, K)
matrix_list[[1]]
matrix_list[[2]]
R
matrix_list[[1]] %*% t(matrix_list[[2]])
