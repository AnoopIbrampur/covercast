# Feature-group ablation

8 walk-forward cutoffs, 79,546 store-days. RMSLE (lower is better); delta_vs_full > 0 means removing the group hurts, with a 95% store-bootstrap interval.

| variant                                |    all |   stores that take bookings |   holiday or next to one |   1-3 days ahead |   8-14 days ahead |   delta_vs_full |   ci_low |   ci_high |   mean4 baseline, all |
|:---------------------------------------|-------:|----------------------------:|-------------------------:|-----------------:|------------------:|----------------:|---------:|----------:|----------------------:|
| full                                   | 0.498  |                      0.5157 |                   0.5704 |           0.5198 |            0.5054 |        nan      | nan      |  nan      |                0.5539 |
| without bookings                       | 0.5065 |                      0.5387 |                   0.5872 |           0.535  |            0.511  |          0.0085 |   0.0066 |    0.0103 |                0.5539 |
| without holidays                       | 0.5015 |                      0.5201 |                   0.6197 |           0.5277 |            0.5091 |          0.0035 |   0.0027 |    0.0043 |                0.5539 |
| without calendar (month, day of month) | 0.4994 |                      0.5177 |                   0.5707 |           0.5203 |            0.507  |          0.0014 |   0.0008 |    0.002  |                0.5539 |
| without same-weekday history           | 0.52   |                      0.5422 |                   0.6117 |           0.5471 |            0.5271 |          0.022  |   0.0198 |    0.0242 |                0.5539 |
| without recent level and volatility    | 0.5073 |                      0.526  |                   0.5849 |           0.53   |            0.5145 |          0.0093 |   0.008  |    0.0105 |                0.5539 |
| without store type, area, age          | 0.4993 |                      0.5167 |                   0.5765 |           0.5215 |            0.5069 |          0.0013 |   0.0007 |    0.0019 |                0.5539 |
| without lead time                      | 0.4981 |                      0.5157 |                   0.5696 |           0.5196 |            0.5056 |          0.0001 |  -0.0001 |    0.0003 |                0.5539 |
