
data {
  int<lower=1> N;                   // number of observations
  int<lower=0, upper=1> y[N];       // binary outcome, willingness to provide
  int<lower=1> K;                   // number of dummy variables
   matrix[N,K] X;                   // matrix of covariate dummy variables
 }
  
parameters {
  vector[K] beta;
}
model {
  // generate our parameter p
  vector[N] p;
  for(i in 1:N){
    p[i] = (X[i]*beta);
  }
  
    // likelihood
  y ~ bernoulli_logit(p);
  
  // priors for betas
  beta ~ normal(0, 1);
}
generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] y_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real y_hat_n = X[n]*beta;
    log_lik[n] = bernoulli_logit_lpmf(y[n] | y_hat_n);
    y_rep[n] = bernoulli_logit_rng(y_hat_n);
  }
}
