# covercast

Forecast restaurant covers 1 to 14 days ahead and score the forecast on what it
costs in labor, not just on forecast error.

Scheduling software usually turns a cover forecast into a labor target. A
forecast that is 10% low and one that is 10% high have the same error, but
they don't cost the same: a short-staffed Friday means overtime, call-ins and
slow tables, while an over-staffed one means idle wages. covercast prices that
difference and forecasts for it directly.

![backtest](docs/backtest.png)

## Results

820 restaurants, 8 weekly cutoffs, 79,546 store-day forecasts for
2017-02-19 to 2017-04-22. Every forecast was made 1 to 14 days ahead,
using only data available at its cutoff.

| forecast | RMSLE | bias | mis-staffing $ per store-week |
|---|---:|---:|---:|
| Same day last week | 0.672 | 0.0% | $431 |
| Average of last 4 same weekdays | 0.554 | -1.9% | $360 |
| **Model** (accuracy-tuned) | **0.498** | **-12.9%** | **$373** |
| Average of 4, best padding | | | $346 |
| Model, best padding | | | $323 |
| **Model, cost-optimal quantile** | | | **$312** |

1. **The model is more accurate.** RMSLE is 0.056 lower than the best
   baseline (95% store-bootstrap interval 0.051 to 0.061), at every lead time
   from 1 to 14 days and in all 8 test weeks.
2. **Staffing to the accurate forecast is still more expensive.** The
   accuracy-tuned model comes in 12.9% under total covers, because minimising
   squared log error targets the typical day, and covers are right-skewed.
   Staffed as-is it costs $373 per store-week, more than the plain
   four-week average ($360).
3. **Training on the right target fixes it.** Trained on the cost-minimising
   quantile, the same features cost $312 per store-week. That is $34 less than
   the best padded baseline (95% interval $30 to $38, about 10%), and it wins
   in all 8 test weeks.

The gain holds where it should, and is largest where simple averages fail:

| segment | store-days | cost saving vs padded 4-week average |
|---|---:|---:|
| stores with bookings in the last 28 days | 27,621 | 13.1% |
| stores without | 51,925 | 7.6% |
| holiday, or the day before or after one | 3,690 | 20% |
| other days | 75,856 | 9.1% |

The 2x cost of a short hour is an assumption, so the whole backtest was rerun
at 1x and 3x (each retrains the quantile model for its own critical ratio):

| short hour costs | staff to | 4-week avg as-is | model as-is | padded 4-week avg | model quantile | saving | weeks won |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x | median | $236 | $220 | $236 | $215 | $21 (9%) | 8/8 |
| 2x | 67th pct | $360 | $373 | $346 | $312 | $34 (10%) | 8/8 |
| 3x | 75th pct | $485 | $527 | $426 | $377 | $48 (11%) | 8/8 |

When short and idle hours cost the same, the accuracy-tuned model is already
the cheapest of the as-is forecasts. Once being short costs more, staffing to
it falls behind the plain average, by more as the asymmetry grows. The quantile
model is the cheapest at every setting. Its calibration is close but not exact:
asked for the 67th percentile, it comes in short on 37% of days rather than 33%.

Full tables: [`artifacts/results/recruit/summary.md`](artifacts/results/recruit/summary.md).

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
is also trained directly on that quantile. Its own padding, used for the
"model, best padding" row, is tuned on held-out earlier cutoffs, because a model
scored on its own training data looks better than it is. To keep the comparison fair, each
baseline gets the best constant padding (for example "last week + 10%"), tuned
on past cutoffs only.

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
