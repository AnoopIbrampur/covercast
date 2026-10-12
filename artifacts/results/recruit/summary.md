# Backtest: Recruit (Kaggle) data

8 weekly cutoffs, target days 2017-02-19 to 2017-04-22, 820 stores, 79,546 store-day forecasts.

## Accuracy

| method    |   rmsle |   mae_covers |   wape |    bias |
|:----------|--------:|-------------:|-------:|--------:|
| last_week |  0.6719 |       9.1264 | 0.4221 |  0.0003 |
| mean4     |  0.5539 |       7.4898 | 0.3464 | -0.0189 |
| model     |  0.498  |       6.9735 | 0.3225 | -0.1296 |

## Staffing cost

Wage $18/h, 4 covers per variable labor hour, short hours cost 2x. Cost-minimising quantile = 0.667.

| method           |   cost_per_store_week |   idle_hours_per_store_week |   short_hours_per_store_week |   short_day_rate |
|:-----------------|----------------------:|----------------------------:|-----------------------------:|-----------------:|
| last_week        |                431.12 |                        7.99 |                         7.98 |             0.47 |
| mean4            |                360.34 |                        6.2  |                         6.91 |             0.48 |
| model            |                373.62 |                        3.65 |                         8.55 |             0.58 |
| last_week_padded |                426.92 |                        9.68 |                         7.02 |             0.44 |
| mean4_padded     |                345.67 |                        8.5  |                         5.35 |             0.4  |
| model_padded     |                318.69 |                        8.87 |                         4.42 |             0.34 |
| model_quantile   |                311.35 |                        7.55 |                         4.87 |             0.37 |
| mean4_dist       |                349.4  |                       10.82 |                         4.3  |             0.32 |
| model_dist       |                310.74 |                        9.27 |                         4    |             0.31 |

## Where it holds

Accuracy vs the average of the last four same weekdays, and labor cost (per store-week) vs that
baseline with its best padding.

| segment                |   store_days |   stores |   rmsle_mean4 |   rmsle_model |   cost_mean4_padded |   cost_model_quantile |   cost_saving |
|:-----------------------|-------------:|---------:|--------------:|--------------:|--------------------:|----------------------:|--------------:|
| all                    |        79546 |      820 |        0.5539 |        0.498  |             345.668 |               311.349 |        0.0993 |
| takes bookings         |        27621 |      298 |        0.5926 |        0.5157 |             409.229 |               354.411 |        0.134  |
| no bookings            |        51925 |      562 |        0.5321 |        0.4883 |             311.857 |               288.443 |        0.0751 |
| holiday or next to one |         3690 |      802 |        0.6635 |        0.5704 |             494.015 |               396.829 |        0.1967 |
| other days             |        75856 |      820 |        0.548  |        0.4942 |             338.452 |               307.191 |        0.0924 |

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
