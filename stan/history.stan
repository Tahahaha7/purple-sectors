// purple-sectors history model: qualifying 2022-2025, to learn driver skill across teams.
// Response y = 100 * log(best lap in a segment), so parameters are in % of lap time.
//   y ~ student_t(nu, alpha[round, segment] + car[season, team, round] + driver[season], sigma)
// Each team-season has its own car (a random walk over that season's rounds, plus track fit);
// driver skill is a random walk over seasons and is centred on each season's grid.

data {
  int<lower=1> N;
  int<lower=1> S;                              // seasons
  int<lower=2> T;                              // teams per season
  int<lower=1> RMAX;                           // most rounds in a season
  array[S] int<lower=1, upper=RMAX> R;          // rounds per season
  int<lower=2> D;                              // drivers
  int<lower=1> A;                              // round x segment baselines
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=1, upper=RMAX> rnd;
  array[N] int<lower=1, upper=T> team;
  array[N] int<lower=1, upper=D> driver;
  array[N] int<lower=1, upper=A> base;
  array[D, S] int<lower=0, upper=1> present;   // driver raced in that season
  vector[A] y_ref;
  vector[N] y;
}

transformed data {
  real sc_t = sqrt(T / (T - 1.0));
  array[S] int n_present;
  for (s in 1:S) n_present[s] = sum(present[:, s]);
}

parameters {
  vector[A] alpha_raw;
  array[S] matrix[T, RMAX] z_car;               // first column: season start; then random-walk steps
  real<lower=0> sigma_car;
  real<lower=0> sigma_dev;                      // per round
  array[S] matrix[T, RMAX] z_nu;
  real<lower=0> sigma_nu;
  matrix[D, S] z_drv;                           // first season level, then season-to-season drift
  real<lower=0> sigma_driver;
  real<lower=0> sigma_drift;                    // per season
  real<lower=0> sigma;
  real<lower=1> nu;
}

transformed parameters {
  vector[A] alpha = y_ref + 5 * alpha_raw;
  array[S] matrix[T, RMAX] car;
  matrix[D, S] skill;
  for (s in 1:S) {
    matrix[T, RMAX] walk;
    walk[:, 1] = sigma_car * z_car[s][:, 1];
    for (r in 2:RMAX) walk[:, r] = walk[:, r - 1] + sigma_dev * z_car[s][:, r];
    for (r in 1:RMAX) {
      vector[T] c = walk[:, r] + sigma_nu * z_nu[s][:, r];
      car[s][:, r] = c - mean(c);                // relative to that round's field
    }
  }
  {
    matrix[D, S] walk;
    walk[:, 1] = sigma_driver * z_drv[:, 1];
    for (s in 2:S) walk[:, s] = walk[:, s - 1] + sigma_drift * z_drv[:, s];
    for (s in 1:S) {                             // relative to that season's grid
      real m = 0;
      for (d in 1:D) if (present[d, s]) m += walk[d, s];
      skill[:, s] = walk[:, s] - m / n_present[s];
    }
  }
}

model {
  alpha_raw ~ std_normal();
  for (s in 1:S) { to_vector(z_car[s]) ~ std_normal(); to_vector(z_nu[s]) ~ std_normal(); }
  to_vector(z_drv) ~ std_normal();
  sigma_car ~ normal(0, 1.2);
  sigma_dev ~ normal(0, 0.2);
  sigma_nu ~ normal(0, 0.5);
  sigma_driver ~ normal(0, 0.3);
  sigma_drift ~ normal(0, 0.1);
  sigma ~ normal(0, 0.5);
  nu ~ gamma(2, 0.1);
  {
    vector[N] mu;
    for (n in 1:N)
      mu[n] = alpha[base[n]] + car[season[n]][team[n], rnd[n]] + skill[driver[n], season[n]];
    y ~ student_t(nu, mu, sigma);
  }
}

generated quantities {
  // skill coming into the next season: last season's level plus one season of drift
  vector[D] skill_next;
  for (d in 1:D) skill_next[d] = skill[d, S] + sigma_drift * normal_rng(0, 1);
}
