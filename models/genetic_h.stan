
data {
  int<lower=1> N;                   // number of observations
  int<lower=0, upper=1> y[N];       // binary outcome, willingness to provide
  int<lower=1> K;                   // number of dummy variables
  int<lower=1> A;                   // number of age groups
  int<lower=1> E;                   // number of education levels
   matrix[N,K] X;                   // matrix of covariate dummy variables
  int<lower=1, upper=A> age[N];     // age group membership
  int<lower=1, upper=E> edu[N];     // education level membership
 }
  
parameters {
  vector[K] beta;
  vector[A] alpha_age;
  vector[E] alpha_edu;
  real <lower=0> sigma_age;
  real <lower=0> sigma_edu;
}
model {
  // generate our parameter p
  vector[N] p;
  for(i in 1:N){
    p[i] = alpha_age[age[i]] + alpha_edu[edu[i]] + (X[i]*beta);
  }
  
    // likelihood
  y ~ bernoulli_logit(p);
  
  // priors for betas
  beta ~ normal(0, 1);
  
  // priors on alpha age
  alpha_age[1] ~ normal(0, 1);
  alpha_age[2:A] ~ normal(alpha_age[1:(A-1)], sigma_age);
  
  // priors for alpha education
  alpha_edu ~ normal(0, sigma_edu);
  
  // priors for variance parameters
  sigma_age ~ normal(0, 1);
  sigma_edu ~ normal(0, 1);
}
generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] y_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real y_hat_n = alpha_age[age[n]] + alpha_edu[edu[n]] + (X[n]*beta);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | y_hat_n);
    y_rep[n] = bernoulli_logit_rng(y_hat_n);
  }
}
