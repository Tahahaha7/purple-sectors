"""The pace model: pace = car + driver + fit, one model for every push lap since 2023.

data      laps → design          python -m purple_sectors.pace.fit --help
simulate  NumPy mirror of stan/pace.stan (prior checks, recovery, lap probabilities)
fit       CmdStan fit, sampler checks, posterior file
export    posterior → site/data/pace.json
"""
