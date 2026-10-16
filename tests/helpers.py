"""Small synthetic seasons and posteriors, so tests run in seconds without network or Stan."""
import numpy as np
import pandas as pd

TEAMS = {"Alpha": ["AAA", "BBB"], "Bravo": ["CCC", "DDD"], "Charlie": ["EEE", "FFF"]}
NAMES = {"AAA": "Ann ALPHA", "BBB": "Ben BETA", "CCC": "Cal GAMMA", "DDD": "Dan DELTA", "EEE": "Eve EPSILON", "FFF": "Fay ZETA"}


def laps_frame(rounds=4, rng=None) -> pd.DataFrame:
    """A tidy fake qualifying lap table with the columns fetch.py writes."""
    rng = rng or np.random.default_rng(0)
    rows, n = [], 1
    for r in range(1, rounds + 1):
        t0 = pd.Timestamp("2026-03-07 05:00", tz="UTC") + pd.Timedelta(weeks=2 * (r - 1))
        for s_i, seg in enumerate(["Q1", "Q2", "Q3"]):
            for team, drivers in TEAMS.items():
                for k, d in enumerate(drivers):
                    for lap in range(3):
                        base = 90 + 0.5 * list(TEAMS).index(team) + 0.1 * k - 0.3 * s_i
                        rows.append(dict(round=r, event=f"Test {r} Grand Prix", location=f"Town{r}", circuit=f"C{r}",
                                         session_key=1000 + r, segment=seg, driver=d, driver_number=n + list(NAMES).index(d),
                                         full_name=NAMES[d], team=team, lap_number=lap + 1,
                                         lap_time=base + abs(rng.normal(0, 0.2)), s1=30, s2=30, s3=30,
                                         date_start=(t0 + pd.Timedelta(minutes=20 * s_i + 4 * lap + k)).isoformat(),
                                         is_pit_out_lap=False, compound="SOFT", tyre_life=lap + 1, deleted=False,
                                         i1_speed=300, i2_speed=280, st_speed=320, air_temp=20, track_temp=30, rainfall=0))
    return pd.DataFrame(rows)


def write_laps(tmp_path, df=None, rounds=4):
    path = tmp_path / "laps.csv"
    (df if df is not None else laps_frame(rounds)).to_csv(path, index=False)
    return path


def posterior(tmp_path, S=400, rounds=4, seed=1, extra=None):
    """A fake fitted posterior with the arrays fit.py saves (seconds-scale effects in % units)."""
    rng = np.random.default_rng(seed)
    drivers = sorted(NAMES); cars = sorted(TEAMS); E, C, D = rounds, len(cars), len(drivers)
    car_of = {d: cars.index(t) for t, ds in TEAMS.items() for d in ds}
    df = laps_frame(rounds)
    arrays = dict(
        theta=rng.normal(np.linspace(-1, 1, C), 0.2, (S, E, C)), nu=rng.normal(0, 0.1, (S, E, C)),
        delta=rng.normal(np.tile([-0.2, 0.2], C), 0.05, (S, D)), tau=rng.normal(-0.4, 0.05, (S, E, 3)),
        gamma=rng.normal(-0.4, 0.05, (S, E)), car_share=rng.uniform(0.95, 0.99, (S, E)),
        **{k: np.abs(rng.normal(m, 0.02, S)) for k, m in [("sigma_car", 1), ("sigma_dev", .05), ("sigma_nu", .3), ("sigma_driver", .15),
                                                          ("sigma", .2), ("pi_bad", .1), ("loss_clean", .28), ("loss_bad", 1.4),
                                                          ("mu_gamma", .45), ("sigma_gamma", .1), ("beta_m", .3), ("beta_a", .03)]},
        drivers=np.array(drivers), cars=np.array(cars), events=np.arange(1, E + 1),
        y=100 * np.log(df["lap_time"].to_numpy()), driver=df["driver"].map(drivers.index).to_numpy(),
        car=df["driver"].map(car_of).to_numpy(), event=df["round"].to_numpy() - 1,
        seg=df["segment"].map({"Q1": 0, "Q2": 1, "Q3": 2}).to_numpy())
    arrays["tau"][:, :, 0] = 0
    arrays.update(extra or {})
    path = tmp_path / "post.npz"
    np.savez(path, **arrays)
    return path, df
