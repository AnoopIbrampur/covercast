# Backtest: synthetic data

8 weekly cutoffs, target days 2017-02-19 to 2017-04-22, 120 stores, 12,119 store-day forecasts.

## Accuracy

| method    |   rmsle |   mae_covers |   wape |    bias |
|:----------|--------:|-------------:|-------:|--------:|
| last_week |  0.466  |       8.2981 | 0.3417 | -0.0092 |
| mean4     |  0.405  |       7.4041 | 0.3049 | -0.0116 |
| model     |  0.3658 |       6.7039 | 0.2761 | -0.0633 |

## Staffing cost

Wage $18/h, 4 covers per variable labor hour, short hours cost 2x. Cost-minimising quantile = 0.667.

| method           |   cost_per_store_week |   idle_hours_per_store_week |   short_hours_per_store_week |   short_day_rate |
|:-----------------|----------------------:|----------------------------:|-----------------------------:|-----------------:|
| last_week        |                395.59 |                        7.07 |                         7.46 |             0.48 |
| mean4            |                354.3  |                        6.23 |                         6.73 |             0.47 |
| model            |                340.98 |                        4.52 |                         7.21 |             0.53 |
| last_week_padded |                386.21 |                        9.52 |                         5.97 |             0.43 |
| mean4_padded     |                341.07 |                        8.96 |                         4.99 |             0.39 |
| model_padded     |                308.75 |                        8    |                         4.57 |             0.38 |
| model_quantile   |                307.01 |                        7.99 |                         4.53 |             0.36 |
| mean4_dist       |                341.05 |                       10.21 |                         4.37 |             0.35 |
| model_dist       |                306.04 |                        8.52 |                         4.24 |             0.35 |

## Where it holds

Accuracy vs the average of the last four same weekdays, and labor cost (per store-week) vs that
baseline with its best padding.

| segment                |   store_days |   stores |   rmsle_mean4 |   rmsle_model |   cost_mean4_padded |   cost_model_quantile |   cost_saving |
|:-----------------------|-------------:|---------:|--------------:|--------------:|--------------------:|----------------------:|--------------:|
| all                    |        12119 |      120 |        0.405  |        0.3658 |             341.065 |               307.009 |        0.0999 |
| takes bookings         |         5959 |       59 |        0.3997 |        0.3532 |             335.156 |               295.479 |        0.1184 |
| no bookings            |         6160 |       61 |        0.41   |        0.3777 |             346.782 |               318.162 |        0.0825 |
| holiday or next to one |          566 |      120 |        0.4763 |        0.4173 |             349.539 |               289.861 |        0.1707 |
| other days             |        11553 |      120 |        0.4012 |        0.3631 |             340.65  |               307.849 |        0.0963 |

## Distribution calibration

coverage@q = share of days at or under the forecast q-quantile (calibrated: equals q).
Pinball loss averaged over 19 levels from 0.05 to 0.95, in covers (lower is better).

|       |   mean_pinball_covers |   coverage@0.1 |   coverage@0.25 |   coverage@0.5 |   coverage@0.75 |   coverage@0.9 |
|:------|----------------------:|---------------:|----------------:|---------------:|----------------:|---------------:|
| mean4 |                 2.755 |          0.097 |           0.242 |          0.485 |           0.732 |          0.889 |
| model |                 2.507 |          0.093 |           0.242 |          0.489 |           0.73  |          0.891 |

## RMSLE by horizon (days ahead)

|   h |   last_week |   mean4 |   model |
|----:|------------:|--------:|--------:|
|   1 |      0.4508 |  0.3959 |  0.3248 |
|   2 |      0.5116 |  0.4518 |  0.3698 |
|   3 |      0.4656 |  0.4056 |  0.3449 |
|   4 |      0.4578 |  0.3964 |  0.3501 |
|   5 |      0.4409 |  0.3895 |  0.3532 |
|   6 |      0.419  |  0.3713 |  0.3441 |
|   7 |      0.3933 |  0.3638 |  0.3417 |
|   8 |      0.4938 |  0.4232 |  0.3845 |
|   9 |      0.5578 |  0.4757 |  0.4281 |
|  10 |      0.4959 |  0.4089 |  0.3818 |
|  11 |      0.4768 |  0.4116 |  0.3787 |
|  12 |      0.4783 |  0.4132 |  0.3848 |
|  13 |      0.4567 |  0.3908 |  0.3713 |
|  14 |      0.4369 |  0.3831 |  0.3604 |
