# Shift-level backtest: Recruit (Kaggle) data

79,546 store-days, 820 stores. Wage $18/h, 15 staff-minutes per cover, short staff-hours cost 2x, minimum crew 1, shifts of 3-8h, 200 demand scenarios.

## Per store-week

| policy       |   total_cost |   scheduled_hours |   short_staff_hours |   excess_over_oracle |
|:-------------|-------------:|------------------:|--------------------:|---------------------:|
| mean4_padded |      1742.5  |             87.66 |                4.57 |                86.09 |
| model_padded |      1737.98 |             88.27 |                4.14 |                81.57 |
| mean4_dist   |      1735.24 |             85.49 |                5.45 |                78.84 |
| model_dist   |      1726.28 |             84.91 |                5.5  |                69.87 |
| oracle       |      1656.41 |             87.64 |                2.19 |                 0    |

Policies: `mean4_padded` = 4-week average + best padding, one scenario; `model_padded` = model mean + best padding, one scenario; `mean4_dist` = optimised over a no-model distribution around the 4-week average; `model_dist` = optimised over the model's calibrated distribution; `oracle` = perfect foresight of the realized hourly need.

## Excess over perfect foresight, by store volume ($ per store-week)

| band           |   store_days |   stores |   mean4_padded |   model_padded |   mean4_dist |   model_dist |   model_dist_cuts_excess_by |
|:---------------|-------------:|---------:|---------------:|---------------:|-------------:|-------------:|----------------------------:|
| <15 covers/day |        36616 |      457 |          27.45 |          23.06 |        25.41 |        21.4  |                        0.22 |
| 15-30          |        30708 |      425 |         115.96 |         109.76 |       107.03 |        95.09 |                        0.18 |
| 30-60          |        11993 |      188 |         183.5  |         181.48 |       165.41 |       147.72 |                        0.19 |
| 60+            |          222 |        5 |         367.93 |         437.03 |       315.99 |       370.34 |                       -0.01 |

Model distribution vs padded 4-week average: -16.22 $/store-week [-18.54, -14.09], 8/8 weeks.

Model distribution vs padded model mean: -11.70 $/store-week [-13.14, -10.31], 8/8 weeks.
