# Benter Project

Inspired by this Bloomberg [article](https://www.bloomberg.com/news/features/2018-05-03/the-gambler-who-cracked-the-horse-racing-code)

### Similar projects:
- [pourquoi/cataclop](https://github.com/pourquoi/cataclop/tree/master/cataclop)

### Usage:
1. Scrape betting website (`scrape_xxx.py`)
2. generate structured data out of it (`generate_xxx.py`)

### State of the project
| Area              | current state         |WIP            |
| ------------------| :--------------------:|--------------:|
|Scraped Website    |PMU Unibet(stopped)    |               |
|Betting type       |Simple_gagnant         | Simple_place  |
|Training           | train on n_horses,permutation, subraces| train on all races, gridsearch, feature selection|
|Winning Horse Model|logistic regression SGD, baselines, skleanrn,|sequential MLP, MLP, transformer, dimension reduction technique|
|Wagering strategy  |Kelly,Most Expected return, baselines, factories| |
|Real time betting  |Real time suggestion   |Real time betting directly on PMU |


---
Appendix
Count python file lines: `git ls-files | grep ".\.py" | xargs wc -l`
Lexicon between French and English: https://www.lexiqueducheval.net/lexique_turf_engl.html
