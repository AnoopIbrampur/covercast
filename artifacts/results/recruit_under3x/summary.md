# Backtest: Recruit (Kaggle) data

8 weekly cutoffs, target days 2017-02-19 to 2017-04-22, 820 stores, 79,546 store-day forecasts.

## Accuracy

| method    |   rmsle |   mae_covers |   wape |    bias |
|:----------|--------:|-------------:|-------:|--------:|
| last_week |  0.6719 |       9.1264 | 0.4221 |  0.0003 |
| mean4     |  0.5539 |       7.4898 | 0.3464 | -0.0189 |
| model     |  0.4978 |       6.9774 | 0.3227 | -0.1286 |

## Staffing cost

Wage $18/h, 4 covers per variable labor hour, short hours cost 3x. Cost-minimising quantile = 0.750.

| method           |   cost_per_store_week |   idle_hours_per_store_week |   short_hours_per_store_week |   short_day_rate |
|:-----------------|----------------------:|----------------------------:|-----------------------------:|-----------------:|
| last_week        |                574.76 |                        7.99 |                         7.98 |             0.47 |
| mean4            |                484.75 |                        6.2  |                         6.91 |             0.48 |
| model            |                527.19 |                        3.67 |                         8.54 |             0.58 |
| last_week_padded |                539.71 |                       13.15 |                         5.61 |             0.36 |
| mean4_padded     |                425.81 |                       11.3  |                         4.12 |             0.31 |
| model_padded     |                399.11 |                       12.14 |                         3.34 |             0.26 |
| model_quantile   |                377.49 |                       10.02 |                         3.65 |             0.28 |

## Where it holds

Accuracy vs the average of the last four same weekdays, and labor cost (per store-week) vs that
baseline with its best padding.

| segment                |   store_days |   stores |   rmsle_mean4 |   rmsle_model |   cost_mean4_padded |   cost_model_quantile |   cost_saving |
|:-----------------------|-------------:|---------:|--------------:|--------------:|--------------------:|----------------------:|--------------:|
| all                    |        79546 |      820 |        0.5539 |        0.4978 |             425.807 |               377.49  |        0.1135 |
| takes bookings         |        27621 |      298 |        0.5926 |        0.5163 |             504.431 |               429.936 |        0.1477 |
| no bookings            |        51925 |      562 |        0.5321 |        0.4877 |             383.984 |               349.593 |        0.0896 |
| holiday or next to one |         3690 |      802 |        0.6635 |        0.5685 |             630.337 |               478.864 |        0.2403 |
| other days             |        75856 |      820 |        0.548  |        0.4941 |             415.858 |               372.559 |        0.1041 |

## RMSLE by horizon (days ahead)

|   h |   last_week |   mean4 |   model |
|----:|------------:|--------:|--------:|
|   1 |      0.7115 |  0.5931 |  0.5184 |
|   2 |      0.7277 |  0.6082 |  0.5257 |
|   3 |      0.7114 |  0.5823 |  0.5162 |
|   4 |      0.677  |  0.5594 |  0.4908 |
|   5 |      0.6523 |  0.5371 |  0.4858 |
|   6 |      0.6065 |  0.4915 |  0.4552 |
|   7 |      0.6033 |  0.4873 |  0.4492 |
|   8 |      0.7087 |  0.5957 |  0.5308 |
|   9 |      0.7337 |  0.6184 |  0.5406 |
|  10 |      0.7204 |  0.5974 |  0.5384 |
|  11 |      0.6881 |  0.5611 |  0.5046 |
|  12 |      0.6691 |  0.5538 |  0.508  |
|  13 |      0.608  |  0.4946 |  0.4624 |
|  14 |      0.6135 |  0.4956 |  0.46   |
