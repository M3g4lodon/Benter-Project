# Benchmark Wagering Strategies

Goals:
 - Expected return > 5%
 - Bets on more than 20% of races
 - Wins more than 10% of the time

## Hypotheses
- Logistic Regression
- data from PMU
- before train date 2018-01-01
- before validation date 2019-07-01
- pari_type E_SIMPLE_GAGNANT

## Results
| Wagering Strategy | Mean Expected Return per race | % bet races | % winning bets |
|:-----------------:|:-----------------------------:|:-------------------------------:|:------------:|
| Random One Horse | -25.0% (std: 3.7)|
| Random All Horses | -23.1% (std: 4.4)|
| Proportional to Pari Mutual Proba | -25.8% (std: 3.8)|
| Proportional to Odds | -23.7% (std: 4.0) |
| Least Risky Horses on odds | -20.8% (std: 4.0)|
| Rickiest Horses on odds | -22.7% (std: 3.9) |
|-|-|-|
| Best expected return| -27.7% (std: 3.4)|
| Best winning Proba | -27.2% (std: 3.5)|
| Proportional to Positive ER | -32.5% (std: 3.4) |
| Proportional to Winning Proba | -24.1% (std: 3.7)
| Best Winning Proba not Max PM proba | -27.7% (std: 3.7)|
| Kelly Criterion | -24.9% (std: 3.9)|
| Min Proba 0.3 Min ER 10| +82.5% (std: 7.74)| 0.15% | 5.6%|
| Min Proba 0.1 Min ER 10| +42.3% (std: 10.6)| 1.2% | 2.1%|
