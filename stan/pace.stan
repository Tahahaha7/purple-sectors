// purple-sectors · pace.stan
//
// One model for every qualifying push lap from 2023 onward, plus each driver's best lap per
// segment for earlier seasons (2022), which keep the teammate network connected.
//   y = 100 * log(lap time), so every parameter is in % of lap time (1% ≈ 0.9 s).
//
//   y  ~ (1 - pi_bad) ExGauss(mu, sigma, 1 / loss_clean) + pi_bad ExGauss(mu, sigma, 1 / loss_bad)
//   mu = baseline[event, segment] + evolution[event] * t
//      + car[season, team, round] + driver[season] + fit[driver, car] + tyres
//   y2 ~ Student-t(nu_best, baseline + car + driver + fit, sigma_best)      best laps, earlier seasons
//
// car     A team in one season is one car. Its pace is a level, plus a random walk over
//         the season's rounds (in √weeks) — the persistent part, `trend` — plus a one-off
//         deviation at each round. Both are centred on the field at every round, so car
//         pace is always relative to the grid.
// driver  Skill relative to that season's grid. A random walk over seasons, starting from
//         a rookie offset in the debut season, with extra movement in seasons two and three.
// fit     One shrunk term per (driver, car) pair that happened. Counterfactuals use it for
//         pairs that were seen and the prior for pairs that were not.

data {
  int<lower=1> N;                           // push laps
  int<lower=1> S;                           // seasons
  int<lower=1> TMAX;                        // most teams in a season
  int<lower=1> RMAX;                        // most rounds in a season
  array[S] int<lower=1, upper=TMAX> T;      // teams in each season
  array[S] int<lower=1, upper=RMAX> R;      // rounds in each season
  int<lower=1> D;                           // drivers
  int<lower=1> P;                           // (driver, car) pairs seen
  int<lower=1> A;                           // baselines = event x segment
  int<lower=1> E;                           // events, all seasons

  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=1, upper=RMAX> rnd;
  array[N] int<lower=1, upper=TMAX> team;
  array[N] int<lower=1, upper=D> driver;
  array[N] int<lower=1, upper=P> pair;
  array[N] int<lower=1, upper=A> base;
  array[N] int<lower=1, upper=E> event;
  vector[N] t;                              // minutes since the segment's mean lap start, / 10
  vector[N] medium;                         // 1 on the medium compound
  vector[N] tyre_age;                       // laps already done on the set
  vector[N] y;

  int<lower=0> N2;                          // best laps from seasons before the lap data
  array[N2] int<lower=1, upper=S> season2;
  array[N2] int<lower=1, upper=RMAX> rnd2;
  array[N2] int<lower=1, upper=TMAX> team2;
  array[N2] int<lower=1, upper=D> driver2;
  array[N2] int<lower=1, upper=P> pair2;
  array[N2] int<lower=1, upper=A> base2;
  vector[N2] y2;

  array[S] vector<lower=0>[RMAX] dt;        // weeks since the previous round (0 for a season's first)
  real<lower=0> dt_bar;                     // mean of the non-zero dt
  vector[A] y_ref;                          // median y per baseline
  array[D, S] int<lower=0, upper=1> present;
  array[D, S] int<lower=0, upper=1> rookie; // debut season
  array[D, S] int<lower=0, upper=1> young;  // second or third season

  int<lower=0, upper=1> prior_only;
  int<lower=0, upper=1> save_loglik;
}

transformed data {
  array[S] vector[RMAX] sqrt_dt;
  array[S] int n_present;
  for (s in 1:S) {
    sqrt_dt[s] = sqrt(dt[s]);
    n_present[s] = 0;
    for (d in 1:D) n_present[s] += present[d, s];
  }
}

parameters {
  vector[A] alpha_raw;                      // baselines
  real mu_gamma;                            // track evolution within a segment
  real<lower=0> sigma_gamma;
  vector[E] z_gamma;

  array[S] matrix[TMAX, RMAX] z_car;        // car: level in column 1, then walk steps
  array[S] matrix[TMAX, RMAX] z_nu;         // car: one-off deviation at each round
  real<lower=0> sigma_car;                  // spread of car levels
  real<lower=0> sigma_car_round;            // how much a car moves from one round to the next
  real<lower=0, upper=1> rho;               // share of that movement that persists

  matrix[D, S] z_drv;                       // driver: level, then steps
  real<lower=0> sigma_driver;
  real<lower=0> sigma_drift;
  real<lower=0> sigma_young;
  real mu_rookie;                           // debut-season offset (positive = slower)
  real mu_young;                            // mean change per season while young

  vector[P] z_fit;
  real<lower=0> sigma_fit;

  real beta_medium;
  real beta_age;

  real<lower=0> sigma;
  real<lower=0> loss_clean;
  real<lower=0> loss_gap;
  real<lower=0, upper=1> pi_bad;

  real<lower=0> sigma_best;                 // best laps: noise
  real<lower=1, upper=100> nu_best;         // best laps: tail weight (a best lap can still be compromised)
}

transformed parameters {
  vector[A] alpha = y_ref + 2 * alpha_raw;
  vector[E] gamma = mu_gamma + sigma_gamma * z_gamma;
  real sigma_dev = sqrt(rho) * sigma_car_round / sqrt(dt_bar);
  real sigma_nu = sqrt(1 - rho) * sigma_car_round;
  array[S] matrix[TMAX, RMAX] trend;        // persistent car pace
  array[S] matrix[TMAX, RMAX] car;          // trend + this round's deviation
  matrix[D, S] skill;
  vector[P] fit = sigma_fit * z_fit;
  real loss_bad = loss_clean + loss_gap;

  for (s in 1:S) {
    matrix[TMAX, RMAX] walk = rep_matrix(0, TMAX, RMAX);
    trend[s] = rep_matrix(0, TMAX, RMAX);
    car[s] = rep_matrix(0, TMAX, RMAX);
    for (tm in 1:T[s]) walk[tm, 1] = sigma_car * z_car[s][tm, 1];
    for (r in 2:R[s])
      for (tm in 1:T[s])
        walk[tm, r] = walk[tm, r - 1] + sigma_dev * sqrt_dt[s][r] * z_car[s][tm, r];
    for (r in 1:R[s]) {
      real m_trend = 0;
      real m_car = 0;
      for (tm in 1:T[s]) {
        trend[s][tm, r] = walk[tm, r];
        car[s][tm, r] = walk[tm, r] + sigma_nu * z_nu[s][tm, r];
        m_trend += trend[s][tm, r];
        m_car += car[s][tm, r];
      }
      m_trend = m_trend / T[s];
      m_car = m_car / T[s];
      for (tm in 1:T[s]) {
        trend[s][tm, r] = trend[s][tm, r] - m_trend;
        car[s][tm, r] = car[s][tm, r] - m_car;
      }
    }
  }
  {
    matrix[D, S] walk;
    for (d in 1:D) {
      for (s in 1:S) {
        if (s == 1 || rookie[d, s] == 1)
          walk[d, s] = rookie[d, s] * mu_rookie + sigma_driver * z_drv[d, s];
        else
          walk[d, s] = walk[d, s - 1] + young[d, s] * mu_young
                       + sqrt(square(sigma_drift) + young[d, s] * square(sigma_young)) * z_drv[d, s];
      }
    }
    for (s in 1:S) {
      real m = 0;
      for (d in 1:D) if (present[d, s] == 1) m += walk[d, s];
      m = m / n_present[s];
      for (d in 1:D) skill[d, s] = walk[d, s] - m;
    }
  }
}

model {
  alpha_raw ~ std_normal();
  mu_gamma ~ normal(0, 0.5);
  sigma_gamma ~ normal(0, 0.3);
  z_gamma ~ std_normal();

  for (s in 1:S) {
    to_vector(z_car[s]) ~ std_normal();
    to_vector(z_nu[s]) ~ std_normal();
  }
  sigma_car ~ normal(0, 1.5);
  sigma_car_round ~ normal(0, 0.6);
  rho ~ beta(2, 4);

  to_vector(z_drv) ~ std_normal();
  sigma_driver ~ normal(0, 0.3);
  sigma_drift ~ normal(0, 0.1);
  sigma_young ~ normal(0, 0.2);
  mu_rookie ~ normal(0, 0.3);
  mu_young ~ normal(0, 0.15);

  z_fit ~ std_normal();
  sigma_fit ~ normal(0, 0.2);

  beta_medium ~ normal(0.5, 0.5);
  beta_age ~ normal(0, 0.1);

  sigma ~ normal(0, 0.5);
  loss_clean ~ normal(0, 0.5);
  loss_gap ~ normal(0, 1);
  pi_bad ~ beta(2, 12);
  sigma_best ~ normal(0, 0.5);
  nu_best ~ gamma(2, 0.1);

  if (prior_only == 0) {
    vector[N] mu;
    for (n in 1:N)
      mu[n] = alpha[base[n]] + gamma[event[n]] * t[n]
              + car[season[n]][team[n], rnd[n]] + skill[driver[n], season[n]]
              + fit[pair[n]] + beta_medium * medium[n] + beta_age * tyre_age[n];
    for (n in 1:N)
      target += log_mix(pi_bad,
                        exp_mod_normal_lpdf(y[n] | mu[n], sigma, inv(loss_bad)),
                        exp_mod_normal_lpdf(y[n] | mu[n], sigma, inv(loss_clean)));
    if (N2 > 0) {
      vector[N2] mu2;
      for (n in 1:N2)
        mu2[n] = alpha[base2[n]] + car[season2[n]][team2[n], rnd2[n]]
                 + skill[driver2[n], season2[n]] + fit[pair2[n]];
      y2 ~ student_t(nu_best, mu2, sigma_best);
    }
  }
}

generated quantities {
  vector[save_loglik ? N : 0] log_lik;
  if (save_loglik == 1) {
    for (n in 1:N) {
      real m = alpha[base[n]] + gamma[event[n]] * t[n]
               + car[season[n]][team[n], rnd[n]] + skill[driver[n], season[n]]
               + fit[pair[n]] + beta_medium * medium[n] + beta_age * tyre_age[n];
      log_lik[n] = log_mix(pi_bad,
                           exp_mod_normal_lpdf(y[n] | m, sigma, inv(loss_bad)),
                           exp_mod_normal_lpdf(y[n] | m, sigma, inv(loss_clean)));
    }
  }
}
