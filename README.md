# covercast

Covercast forecasts how many guests a restaurant will have over the next 14 days, turns the forecast into a shift schedule, and checks the schedule by replaying the days that actually happened.

I built it to see whether a better forecast changes what a restaurant pays for labor. The short answer is yes, a little, and mostly for reasons that have nothing to do with fancier modeling. The details are below, including the parts that didn't work.

Demo: https://anoopibrampur.github.io/covercast/ (runs on synthetic data, see below)

## What is real and what isn't

Real:
- The visitor counts and reservations come from the Recruit Restaurant Visitor Forecasting dataset on Kaggle. It covers 829 Japanese restaurants from January 2016 to April 2017, about 250,000 store-days and 120,000 bookings.
- Weather comes from the Open-Meteo archive for each restaurant's location.
- Every forecast accuracy number in this README is measured on real restaurants, forecasting forward in time.

Not real:
- The public demo uses a fake restaurant group I generated, because Kaggle's rules don't allow republishing the real data. The page says "synthetic data" on it. The real-data version runs locally.
- The hourly arrival pattern is inferred. The dataset only has daily totals, so I estimated each restaurant's dinner rush from the times its bookings were for, then split each real daily total into hours with random draws. Everything about shifts and hourly staffing rests on that guess.
- Wages, staff time per cover, and the cost of being short-staffed are my assumptions. Change them and the dollar figures change.
- The weather is observed weather, not a forecast, because forecasts from 2016 and 2017 aren't archived.

Other limits: the test window is eight weeks, February to April 2017. These are small Japanese restaurants, not American chains. Nobody in the restaurant business has looked at this yet.

## Results

I forecast every restaurant 1 to 14 days ahead at eight weekly cutoffs, using only what was known at each cutoff, and scored 79,546 store-days. Accuracy is RMSLE (error on a log scale, lower is better).

| forecast | RMSLE |
|---|---:|
| same day last week | 0.672 |
| average of the last 4 same weekdays | 0.554 |
| weekday exponential smoothing, tuned on earlier weeks | 0.514 |
| gradient boosted model | 0.498 |

The headline I started with was that the model beats the 4-week average by 0.056. That number is true and misleading, because the 4-week average is a weak opponent. A small smoothing model with three parameters gets most of the way there. Against that, the boosted model's edge is 0.016 (95% interval 0.012 to 0.021, resampling restaurants). It wins, but narrowly.

### Where the model's edge comes from

I removed one group of inputs at a time and retrained at every cutoff. The number is how much worse RMSLE gets without the group.

| removed | worse by | where it hurts most |
|---|---:|---|
| same-weekday history | 0.022 | everywhere |
| recent level and volatility | 0.009 | everywhere |
| bookings | 0.009 | 0.023 at restaurants that take bookings |
| holidays | 0.004 | 0.049 on holidays and the days next to them |
| month and day of month | 0.001 | |
| store type, area and age | 0.001 | |
| lead time | 0.000 | the interval includes zero |

Most of the model is weekday levels done well, which smoothing nearly matches. The part smoothing can't do is use bookings and holidays, and those two roughly account for the 0.016 gap.

### Things that didn't help

- A booking pickup model. This is the standard airline and hotel method: forecast equals what is already booked plus what usually arrives later. On its own it is worse than the boosted model at restaurants that take bookings (RMSLE higher by 0.064). Added as an input, it changes nothing (difference +0.0000, interval -0.0004 to +0.0004). The trees had already learned the same signal from the booking inputs.
- Tuning. I tried 16 random settings plus the default, scored on four weeks before the test window. 13 of the 16 landed within 0.002 of the default. The best one improved test RMSLE from 0.4980 to 0.4967. That is real but too small to matter.
- Weather. It moves demand a little: rain during dinner hours, snow and unusually cold days each cut covers by about 3% against the restaurant's usual. As a forecasting input it adds almost nothing. With a perfect weather forecast the improvement is 0.0004, and with weather only one to three days ahead it is 0.0003. On rainy evenings it is 0.005 at best. I left it out of the model.

Together these say the model doesn't need better algorithms. It needs information it doesn't have.

### Where the error is

Roughly 3.6% of store-days came in under a third of the restaurant's usual level. They aren't entirely random (restaurants with no such days in the previous eight weeks had them 1.1% of the time, and restaurants where over 8% of recent days were like that had them 4.7% of the time), but I couldn't predict which days from anything in the data. Days with three covers or fewer are 7.5% of the data and 29% of the squared error. Part of the accuracy score is noise that no model here can touch, so I put more weight on cost than on RMSLE.

## Cost, not just accuracy

A forecast that is 10% low and one that is 10% high have the same error, but they don't cost the same. Being short-staffed means overtime, call-ins and slow tables. Being over-staffed means idle wages. I priced short hours at twice the cost of idle ones (an assumption), which makes the cheapest staffing level the 67th percentile of demand, not the average.

The most accurate model has a problem here. It comes in 13% under total covers, because a model trained on squared log error aims at the typical day, and covers are right-skewed. Staffed as-is it costs more than the plain 4-week average.

| forecast, staffed directly | wasted labor cost per store-week |
|---|---:|
| 4-week average | $360 |
| boosted model, as-is | $374 |
| 4-week average, best constant padding | $346 |
| boosted model, best constant padding | $319 |
| boosted model, trained for the 67th percentile | $311 |

The percentile model saves $34 per store-week against the padded average (95% interval $31 to $38), about 10%, and wins all eight test weeks. The saving is $21 if short hours cost the same as idle ones and $48 if they cost three times as much.

One gap: I compared the cost against the padded 4-week average, not against padded smoothing. Given the accuracy results, the saving against smoothing is probably smaller. I haven't run that comparison.

The saving is largest around holidays (20%) and at restaurants that take bookings (13%). It is still 7.5% at restaurants that don't.

## Shifts

Real schedules come in whole people working blocks of 3 to 8 hours, with someone on the floor whenever the doors are open. For each store-day I built the cheapest schedule against 200 demand scenarios drawn from the forecast, then charged it for the simulated hourly arrivals of the real day. A schedule built knowing the real day is the floor.

| schedule built from | cost above the floor, per store-week |
|---|---:|
| padded 4-week average | $86 |
| padded model mean | $82 |
| the 4-week average's own error range | $79 |
| the model's forecast range | $70 |

The model's range saves $16 per store-week (interval $14 to $19) against the padded average, in all eight weeks. That is about half the saving the simple daily cost model suggested, because minimum crews and whole-person shifts absorb a lot of forecast error. Remember that these hourly arrivals are simulated, so this is the weakest result here.

Restaurants under 15 covers a day barely need a forecast, since one person covers the evening either way. For the five busiest restaurants (over 60 a day) the model only ties the average, and the average is the more accurate forecast for them.

## Two things that went wrong

My first forecast ranges were calibrated on a model trained four to eight weeks before the days it was checked on, while real forecasts are made zero to two weeks out. These stores were growing, so the old model's errors leaned high and the medians were five points off (55% of days fell under the median instead of 50%). Calibrating on forecasts made the way real ones are fixed most of it. Coverage at the 10, 25, 50, 75 and 90% levels is now 10.8, 26.8, 52.8, 77.0 and 90.9%.

The single global model also pulled the busiest restaurants toward the middle, because most restaurants in the training data are small. It under-forecast the busiest ones by 31%. Predicting each day relative to the restaurant's own usual level cut that to 13% and changed nothing for the smaller restaurants.

## How it works

- Walk-forward testing. At each evaluation cutoff the model trains only on earlier cutoffs whose whole 14-day horizon had already happened, so no training target comes after the forecast being judged.
- Bookings as of the cutoff. A booking made the day after the cutoff is invisible to that forecast even though it is in the file. A test replaces every visit and booking after the cutoff with noise and checks that no feature changes. I confirmed it fails when I plant a one-day leak.
- A missing day means the store was closed. It is never a target and never counted as zero.
- The model is one gradient boosted model (scikit-learn HistGradientBoosting) over all stores and lead times, predicting log covers relative to the store's own same-weekday level. Inputs are recent levels, same-weekday history, holidays, store type and area, bookings known so far, and how those compare with what the store usually has booked at the same lead time.
- The forecast range is the center plus a scale times a standardized error. A second model predicts the scale. The shape of the errors comes from recent forecasts that neither model trained on (split conformal).
- Shifts come from an integer program solved with HiGHS. Because staffing is a whole number of people, the expected shortfall at each level can be precomputed, so the program doesn't grow with the number of scenarios (about 14 ms per solve). A test checks it against the textbook formulation.

| assumption | default | flag |
|---|---|---|
| wage | $18/h | --wage |
| covers per labor hour | 4 | --covers-per-hour |
| short hour cost vs idle hour | 2x | --under-multiplier |
| minimum crew | 1 | covercast schedule --min-crew |

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

On synthetic data, with no download:

```bash
.venv/bin/covercast synth
.venv/bin/covercast backtest --data-dir artifacts/synthetic
.venv/bin/covercast schedule artifacts/results/synthetic --data-dir artifacts/synthetic
.venv/bin/covercast site artifacts/results/synthetic --data-dir artifacts/synthetic
```

On the real data, accept the competition rules on Kaggle, unzip the CSVs into artifacts/raw/, and run:

```bash
.venv/bin/covercast backtest
.venv/bin/covercast schedule artifacts/results/recruit
.venv/bin/python scripts/ablate.py
.venv/bin/python scripts/benchmarks.py
.venv/bin/python scripts/tune.py
.venv/bin/pip install -e ".[weather]" && .venv/bin/python -m covercast.weather
.venv/bin/python scripts/weather_experiment.py
```

The backtest takes about 10 minutes and the shift replay about 13 on a laptop. Tables go to artifacts/results/. The Kaggle data and anything containing its visitor counts stay out of git.

## What I'd want next

The most useful thing would be real sales by hour from one restaurant. It would replace my guessed arrival pattern, let me test shift results on actual hours, and show whether the accuracy gains carry over to a different kind of restaurant. Without it, the daily results are solid and the shift results are a model of a model.

## License

MIT
