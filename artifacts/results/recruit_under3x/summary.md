# Backtest: Recruit (Kaggle) data

8 weekly cutoffs, target days 2017-02-19 to 2017-04-22, 820 stores, 79,546 store-day forecasts.

## Accuracy

| method    |   rmsle |   mae_covers |   wape |    bias |
|:----------|--------:|-------------:|-------:|--------:|
| last_week |  0.6719 |       9.1264 | 0.4221 |  0.0003 |
| mean4     |  0.5539 |       7.4898 | 0.3464 | -0.0189 |
| model     |  0.498  |       6.9735 | 0.3225 | -0.1296 |

## Staffing cost

Wage $18/h, 4 covers per variable labor hour, short hours cost 3x. Cost-minimising quantile = 0.750.

| method           |   cost_per_store_week |   idle_hours_per_store_week |   short_hours_per_store_week |   short_day_rate |
|:-----------------|----------------------:|----------------------------:|-----------------------------:|-----------------:|
| last_week        |                574.76 |                        7.99 |                         7.98 |             0.47 |
| mean4            |                484.75 |                        6.2  |                         6.91 |             0.48 |
| model            |                527.56 |                        3.65 |                         8.55 |             0.58 |
| last_week_padded |                539.71 |                       13.15 |                         5.61 |             0.36 |
| mean4_padded     |                425.81 |                       11.3  |                         4.12 |             0.31 |
| model_padded     |                393.44 |                       11.53 |                         3.44 |             0.27 |
| model_quantile   |                377.76 |                        9.95 |                         3.68 |             0.29 |
| mean4_dist       |                432.94 |                       14.47 |                         3.19 |             0.24 |
| model_dist       |                377.53 |                       12.19 |                         2.93 |             0.23 |

## Where it holds

Accuracy vs the average of the last four same weekdays, and labor cost (per store-week) vs that
baseline with its best padding.

| segment                |   store_days |   stores |   rmsle_mean4 |   rmsle_model |   cost_mean4_padded |   cost_model_quantile |   cost_saving |
|:-----------------------|-------------:|---------:|--------------:|--------------:|--------------------:|----------------------:|--------------:|
| all                    |        79546 |      820 |        0.5539 |        0.498  |             425.807 |               377.764 |        0.1128 |
| takes bookings         |        27621 |      298 |        0.5926 |        0.5157 |             504.431 |               428.199 |        0.1511 |
| no bookings            |        51925 |      562 |        0.5321 |        0.4883 |             383.984 |               350.935 |        0.0861 |
| holiday or next to one |         3690 |      802 |        0.6635 |        0.5704 |             630.337 |               486.429 |        0.2283 |
| other days             |        75856 |      820 |        0.548  |        0.4942 |             415.858 |               372.478 |        0.1043 |

## Distribution calibration

coverage@q = share of days at or under the forecast q-quantile (calibrated: equals q).
Pinball loss averaged over 19 levels from 0.05 to 0.95, in covers (lower is better).

|       |   mean_pinball_covers |   coverage@0.1 |   coverage@0.25 |   coverage@0.5 |   coverage@0.75 |   coverage@0.9 |
|:------|----------------------:|---------------:|----------------:|---------------:|----------------:|---------------:|
| mean4 |                 2.929 |          0.106 |           0.259 |          0.512 |           0.757 |          0.901 |
| model |                 2.578 |          0.108 |           0.268 |          0.528 |           0.77  |          0.909 |

## RMSLE by horizon (days ahead)

|   h |   last_week |   mean4 |   model |
|----:|------------:|--------:|--------:|
|   1 |      0.7115 |  0.5931 |  0.5177 |
|   2 |      0.7277 |  0.6082 |  0.5255 |
|   3 |      0.7114 |  0.5823 |  0.5164 |
|   4 |      0.677  |  0.5594 |  0.4918 |
|   5 |      0.6523 |  0.5371 |  0.4853 |
|   6 |      0.6065 |  0.4915 |  0.4553 |
|   7 |      0.6033 |  0.4873 |  0.4507 |
|   8 |      0.7087 |  0.5957 |  0.5298 |
|   9 |      0.7337 |  0.6184 |  0.5402 |
|  10 |      0.7204 |  0.5974 |  0.5392 |
|  11 |      0.6881 |  0.5611 |  0.5048 |
|  12 |      0.6691 |  0.5538 |  0.5083 |
|  13 |      0.608  |  0.4946 |  0.4629 |
|  14 |      0.6135 |  0.4956 |  0.4605 |
