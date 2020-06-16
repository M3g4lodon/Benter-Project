# Benchmark Wagering Strategies

## Hypotheses
- Logistic Regression
- data from PMU
- before train date 2018-01-01
- before validation date 2019-07-01
- pari_type E_SIMPLE_GAGNANT

## Results
| Wagering Strategy | Mean Expected Return per race | Average # races between winning | Max Drawdown |
|:-----------------:|:-----------------------------:|:-------------------------------:|:------------:|
| Random One Horse | -27.6% (std: 3.75)|
| Best expected return| -32.2% (std: 5.64)|
| Best winning Proba | -12.4% (std: 2.11)|
| Proportional to Pari Mutual Proba | -12.8% (std: 0.63)|
| Best Winning Proba not Max PM proba | -13.3% (std: 2.61)|
| Kelly Criterion | -29.6% (std: 4.09)|
| Min Proba 0.3 Min ER 10| +64.3% (std: 7.34)|
| Min Proba 0.1 Min ER 10| +39.5% (std: 10.5)|
| Min Proba 0.3 Min ER 6| +6.0% (std: 5.0)|
