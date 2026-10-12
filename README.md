# covercast

Forecast restaurant covers 1 to 14 days ahead and score the forecast on what it
costs in labor, not just on forecast error.

Scheduling software usually turns a cover forecast into a labor target. A
forecast that is 10% low and one that is 10% high have the same error, but
they don't cost the same: a short-staffed Friday means overtime, call-ins and
slow tables, while an over-staffed one means idle wages. covercast prices that
difference and forecasts for it directly.

> **Status:** the pipeline, the leakage tests and a synthetic dry run are done.
> Results on the real Recruit data are next. Nothing below is a claim about
> real restaurants until that run is in.

## The question

At each weekly cutoff, for every active restaurant, forecast covers for each
of the next 14 days (about how far ahead a schedule is written). Then:

1. **Accuracy:** does a model beat what managers actually do, "same day last
   week" and "average of the last four same weekdays"?
2. **Labor cost:** once each forecast becomes staffed hours, which one wastes
   the fewest dollars?

## Data

[Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting)
(Kaggle, 2017): daily visitor counts for 829 restaurants in Japan from
Jan 2016 to Apr 2017, plus bookings from two reservation sites with the time
each booking was made. The data isn't redistributable, so it isn't in this repo.

## Method

- **Walk-forward.** Weekly cutoffs. At each evaluation cutoff the model is
  retrained only on earlier cutoffs whose whole 14-day horizon had already
  happened, so no training target postdates the forecast being judged.
- **Bookings as of the cutoff.** A booking made the day after the cutoff is
  invisible to that forecast even though it's in the file. Getting this wrong
  is the easy way to overstate how much reservations help.
  [`test_features_ignore_everything_after_cutoff`](tests/test_features.py)
  replaces every visit and booking after the cutoff with noise and checks
  that no feature changes. Planting a one-day booking leak makes it fail.
- **Closed is not zero.** A missing day means the store was shut. It is never
  a target, and never averaged in as a zero.
- **Model.** One gradient-boosted model (scikit-learn `HistGradientBoosting`)
  across all stores and lead times, trained on log covers. Features are
  recent levels, same-weekday history, holidays, store type and area, bookings
  known so far, and how those bookings compare with what this store usually
  has at the same lead time.

### From covers to dollars

Only the labor that scales with covers depends on the forecast. Staffing for
F covers when A show up leaves |F - A| / productivity hours idle or short:

| assumption | default | flag |
|---|---|---|
| wage | $18/h | `--wage` |
| covers per variable labor hour | 4 | `--covers-per-hour` |
| cost of a short hour vs an idle one | 2x | `--under-multiplier` |

With short hours costing 2x, the cost-minimising staffing level is the 2/3
quantile of demand (the newsvendor critical ratio), not the mean. So the model
is also trained directly on that quantile. To keep the comparison fair, each
baseline gets the best constant padding (for example "last week + 10%"), tuned
on past cutoffs only.

One more reason the mean is the wrong target: a model trained for the
competition metric (squared log error) forecasts the *typical* day, which for
right-skewed covers sits below the average. On synthetic data it comes in 6.5%
under total covers, exactly the wrong direction for staffing.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Dry run on synthetic data in the same schema (no download needed):

```bash
.venv/bin/covercast synth
.venv/bin/covercast backtest --data-dir artifacts/synthetic
```

Real data: accept the competition rules on Kaggle, download the data, unzip
the CSVs into `artifacts/raw/`, then:

```bash
.venv/bin/covercast backtest
```

Results go to `artifacts/results/<dataset>/`: `summary.md`, `summary.json`,
`predictions.csv.gz` and `backtest.png`.

## Limits

- Recruit gives daily covers, not hourly, so labor targets here are per day.
  Splitting a day into shifts needs ticket-time data from a POS.
- Productivity and the cost of a short hour are assumptions, and they are
  flags so they can be changed.
- These are small Japanese restaurants, and US multi-unit groups may behave
  differently. The method carries over; the numbers might not.

## License

MIT
