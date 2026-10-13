# Weather experiment

Observed weather from the Open-Meteo archive. *ceiling* = target-day weather at every lead time (a perfect weather forecast); *realistic* = only 3 or fewer days ahead.

## Does weather move demand? (covers relative to the store's usual level, pre-test history)

|                               |   days |   median covers vs usual |
|:------------------------------|-------:|-------------------------:|
| dry (<1 mm)                   |  12202 |                    1.02  |
| rain 1-10 mm                  |   5852 |                    1.009 |
| rain >10 mm                   |   2883 |                    0.995 |
| evening rain >=3 mm           |   2104 |                    0.966 |
| snow                          |    610 |                    0.968 |
| much colder than usual (<-5C) |   1084 |                    0.972 |
| much warmer than usual (>+5C) |    610 |                    1.02  |

## Forecast accuracy (RMSLE)

|                                      |   store_days |   base |   ceiling |   realistic |
|:-------------------------------------|-------------:|-------:|----------:|------------:|
| all                                  |        79546 | 0.498  |    0.4976 |      0.4977 |
| 1-3 days ahead                       |        15621 | 0.5198 |    0.5191 |      0.519  |
| rainy days (>=5 mm)                  |        14234 | 0.5175 |    0.5158 |      0.5161 |
| rainy evenings (>=3 mm, 17-22h)      |         5511 | 0.514  |    0.5092 |      0.5113 |
| unusual temperature (|anomaly| > 5C) |         7111 | 0.5188 |    0.5182 |      0.5181 |

| comparison | RMSLE difference | 95% interval |
|---|---:|---:|
| ceiling vs base | -0.0004 | -0.0009 to +0.0000 |
| realistic vs base | -0.0003 | -0.0006 to -0.0001 |
| realistic vs base, 1-3 days ahead | -0.0008 | -0.0015 to -0.0001 |
| ceiling vs base, rainy days | -0.0017 | -0.0032 to -0.0002 |
