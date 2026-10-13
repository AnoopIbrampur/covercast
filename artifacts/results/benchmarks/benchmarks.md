# Benchmarks

Same 8 walk-forward test weeks as the backtest, 79,546 store-days. Smoothing tuned on the 4 cutoffs before them: {'alpha': 0.1, 'alpha_dev': 0.1, 'phi': 0.9}.
Pickup applies only to stores that take bookings (elsewhere it falls back to the 4-week average).

|                                         |   mean4 |   smooth |   model |   model_pickup |   pickup |
|:----------------------------------------|--------:|---------:|--------:|---------------:|---------:|
| all                                     |  0.5539 |   0.5139 |  0.498  |         0.498  |   0.5383 |
| excluding collapse days (<35% of usual) |  0.4582 |   0.434  |  0.4219 |         0.422  |   0.4435 |
| stores that take bookings               |  0.5926 |   0.5514 |  0.5157 |         0.5156 |   0.5801 |

| comparison | RMSLE difference | 95% interval |
|---|---:|---:|
| model vs smoothing | -0.0159 | -0.0206 to -0.0116 |
| model+pickup vs model | +0.0000 | -0.0004 to +0.0004 |
| pickup vs model (booking stores) | +0.0644 | +0.0387 to +0.1098 |
| model+pickup vs model (booking stores) | -0.0002 | -0.0011 to +0.0008 |
