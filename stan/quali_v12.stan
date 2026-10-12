// purple-sectors v1.2: as v1.1, with a two-component execution model:
//   y ~ (1 - pi) * ExGauss(mu, sigma, 1/loss_clean) + pi * ExGauss(mu, sigma, 1/loss_bad)
//   pi = share of compromised push laps (traffic, mistake, yellow); loss_bad > loss_clean.
// purple-sectors v1.1: as v1, plus track evolution *within* each segment:
//   tau_e(t) = kappa_{e,s} + gamma_e * t,  t = minutes from segment's mean lap start / 10
// purple-sectors v1: qualifying pace = event + track evolution + car + driver + tyres
// Response y = 100 * log(lap time), so every parameter is in "% of lap time".
// Requires CmdStan >= 2.36 (sum_to_zero_vector).

functions {
  vector linear_predictor(vector alpha, matrix tau, vector gamma, array[] vector theta,
                          array[] vector nu, vector delta, real beta_m, real beta_a,
                          array[] int event, array[] int seg, array[] int car,
                          array[] int driver, vector medium, vector tyre_age, vector t_seg) {
    int N = size(event);
    vector[N] mu;
    for (n in 1:N) {
      int e = event[n];
      int c = car[n];
      mu[n] = alpha[e] + tau[e, seg[n]] + gamma[e] * t_seg[n]
              + theta[e][c] + nu[e][c] + delta[driver[n]];
    }
    return mu + beta_m * medium + beta_a * tyre_age;
  }
}

data {
  int<lower=1> N;                             // push laps
  int<lower=2> D;                             // drivers
  int<lower=2> C;                             // cars (teams)
  int<lower=1> E;                             // events, in date order
  array[N] int<lower=1, upper=D> driver;
  array[N] int<lower=1, upper=C> car;
  array[N] int<lower=1, upper=E> event;
  array[N] int<lower=1, upper=3> seg;         // 1 = Q1, 2 = Q2, 3 = Q3
  vector<lower=0, upper=1>[N] medium;         // 1 if medium compound
  vector<lower=0>[N] tyre_age;                // laps beyond fresh
  vector<lower=0>[E] dt;                      // weeks since previous event (dt[1] unused)
  vector[N] t_seg;                            // (lap start - segment mean) in 10-min units
  vector[E] y_ref;                            // event median of y, for centering
  vector[N] y;                                // 100 * log(lap time)
  int<lower=0, upper=1> prior_only;           // 1 = ignore likelihood (prior predictive)
}

transformed data {
  real sc_car = sqrt(C / (C - 1.0));          // keeps marginal sd = sigma under sum-to-zero
  real sc_drv = sqrt(D / (D - 1.0));
}

parameters {
  vector[E] alpha_raw;                        // event baselines
  matrix[E, 2] kappa_raw;                     // Q1->Q2, Q2->Q3 steps
  vector[2] m;
  real<lower=0> sigma_kappa;

  vector[E] gamma_raw;                        // within-segment evolution per 10 min
  real mu_gamma;
  real<lower=0> sigma_gamma;

  array[E] sum_to_zero_vector[C] z_car;       // car start + random-walk innovations
  real<lower=0> sigma_car;
  real<lower=0> sigma_dev;

  array[E] sum_to_zero_vector[C] z_nu;        // car x track fit
  real<lower=0> sigma_nu;

  sum_to_zero_vector[D] z_driver;
  real<lower=0> sigma_driver;

  real beta_m;
  real beta_a;

  real<lower=0> sigma;                        // lap noise
  real<lower=0, upper=1> pi_bad;              // share of compromised push laps
  real<lower=0.01> loss_clean;                // mean loss on a clean lap
  real<lower=0> loss_gap;                     // extra mean loss on a compromised lap
}

transformed parameters {
  vector[E] alpha = y_ref + 5 * alpha_raw;
  matrix[E, 3] tau;
  array[E] vector[C] theta;                   // car pace trend (development)
  array[E] vector[C] nu;                      // car x track fit
  vector[D] delta = sigma_driver * z_driver;  // driver skill
  vector[E] gamma = mu_gamma + sigma_gamma * gamma_raw;
  real loss_bad = loss_clean + loss_gap;

  for (e in 1:E) {
    real k2 = m[1] + sigma_kappa * kappa_raw[e, 1];
    real k3 = m[2] + sigma_kappa * kappa_raw[e, 2];
    tau[e] = [0, k2, k2 + k3];
    nu[e] = sigma_nu * z_nu[e];
  }
  theta[1] = sigma_car * z_car[1];
  for (e in 2:E)
    theta[e] = theta[e - 1] + sigma_dev * sqrt(dt[e]) * z_car[e];
}

model {
  alpha_raw ~ std_normal();
  to_vector(kappa_raw) ~ std_normal();
  for (e in 1:E) {
    z_car[e] ~ normal(0, sc_car);
    z_nu[e] ~ normal(0, sc_car);
  }
  z_driver ~ normal(0, sc_drv);

  m ~ normal(-0.5, 0.3);
  gamma_raw ~ std_normal();
  mu_gamma ~ normal(-0.3, 0.3);      // per 10 min; absorbs rubbering + final-run effect
  sigma_gamma ~ normal(0, 0.2);
  sigma_kappa ~ normal(0, 0.3);
  sigma_car ~ normal(0, 1.2);   // prior predictive: bounded by 107% rule
  sigma_dev ~ normal(0, 0.25);
  sigma_nu ~ normal(0, 0.5);
  sigma_driver ~ normal(0, 0.3); // driver worth > 1.5% is implausible
  beta_m ~ normal(0.7, 0.5);
  beta_a ~ normal(0.05, 0.05);
  sigma ~ normal(0, 0.5);
  pi_bad ~ beta(2, 6);
  loss_clean ~ normal(0, 0.3);
  loss_gap ~ normal(0, 1.5);

  if (!prior_only) {
    vector[N] mu = linear_predictor(alpha, tau, gamma, theta, nu, delta, beta_m, beta_a,
                                    event, seg, car, driver, medium, tyre_age, t_seg);
    for (n in 1:N)
      target += log_mix(pi_bad,
                        exp_mod_normal_lpdf(y[n] | mu[n], sigma, inv(loss_bad)),
                        exp_mod_normal_lpdf(y[n] | mu[n], sigma, inv(loss_clean)));
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] p_bad;                            // P(lap n was compromised | data)
  vector[E] car_share;                        // Var(car) / (Var(car) + Var(driver))
  {
    vector[N] mu = linear_predictor(alpha, tau, gamma, theta, nu, delta, beta_m, beta_a,
                                    event, seg, car, driver, medium, tyre_age, t_seg);
    for (n in 1:N) {
      real lp_bad = log(pi_bad) + exp_mod_normal_lpdf(y[n] | mu[n], sigma, inv(loss_bad));
      real lp_ok = log1m(pi_bad) + exp_mod_normal_lpdf(y[n] | mu[n], sigma, inv(loss_clean));
      log_lik[n] = log_sum_exp(lp_bad, lp_ok);
      p_bad[n] = exp(lp_bad - log_lik[n]);
      y_rep[n] = exp_mod_normal_rng(mu[n], sigma,
                                    inv(bernoulli_rng(pi_bad) ? loss_bad : loss_clean));
    }
  }
  for (e in 1:E) {
    real v_car = variance(theta[e] + nu[e]);
    car_share[e] = v_car / (v_car + variance(delta));
  }
}
