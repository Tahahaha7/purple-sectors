import pandas as pd

import purple_sectors.history as H


def _race(season, rnd, entries):
    return {"season": str(season), "round": str(rnd), "raceName": f"GP {rnd}", "date": "2024-03-01",
            "Circuit": {"circuitId": f"c{rnd}", "Location": {"locality": "Town"}},
            "QualifyingResults": [{"position": str(i + 1), "Q1": "1:30.123", "Q2": "1:29.900" if i < 15 else "", "Q3": "1:29.5" if i < 10 else None,
                                   "Driver": {"driverId": d.lower(), "code": d, "givenName": "A", "familyName": d.title(), "permanentNumber": str(i + 1)},
                                   "Constructor": {"constructorId": t.lower(), "name": t}} for i, (d, t) in enumerate(entries)]}


def test_parse_and_paging():
    races = [_race(2024, r, [("AAA", "Red"), ("BBB", "Red"), ("CCC", "Blue")]) for r in range(1, 4)]
    pages = [races[:2], races[2:]]

    class S:
        calls = 0
        def get(self, url, params=None, timeout=None):
            S.calls += 1
            i = params["offset"] // H.PAGE
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self): return {"MRData": {"total": str(2 * H.PAGE), "RaceTable": {"Races": pages[i]}}}
            return R()
    H.MIN_INTERVAL = 0
    df = H.fetch_season(2024, S())
    assert S.calls == 2 and len(df) == 9 and set(df["round"]) == {1, 2, 3}
    assert df["q1"].iloc[0] == 90.123 and df["q3"].notna().all()           # all three reached Q3 in the fake data
    assert H.to_sec("") is None and H.to_sec(None) is None and H.to_sec("1:29.5") == 89.5


def test_network_links_drivers_through_teammates():
    hist = pd.DataFrame([dict(season=s, round=1, team=t, driver=d, name=d) for s, t, d in [
        (2024, "Red", "AAA"), (2024, "Red", "BBB"), (2025, "Blue", "BBB"), (2025, "Blue", "CCC"),
        (2025, "Green", "DDD"), (2025, "Green", "EEE")]])
    laps = pd.DataFrame([dict(round=1, team=t, driver=d, full_name=d) for t, d in [("Red", "AAA"), ("Red", "CCC"), ("Green", "DDD"), ("Green", "EEE")]])
    net = H.network(hist, laps)
    assert not net["connected"]                                   # Green (DDD, EEE) is an island
    assert sorted(map(sorted, net["components"])) == [["AAA", "CCC"], ["DDD", "EEE"]]
    edges = {(e["a"], e["b"], e["season"]) for e in net["edges"]}
    assert ("AAA", "BBB", 2024) in edges and ("BBB", "CCC", 2025) in edges and ("AAA", "CCC", 2026) in edges
