## NEAT Flappy Birds

Basic A.I. that plays Flappy birds using the [NEAT-python](https://neat-python.readthedocs.io/en/latest/index.html#) module

I used this project by [Tech with Tom](https://www.youtube.com/playlist?list=PLzMcBGfZo4-lwGZWXz5Qgta_YNX3_vLS2) as a practical implementation of some of
the reinforced machine learning techniques I've been learning. The NEAT-python module is a powerful tool for ML and I found it very ergonomic from a data science point of view. 
It allowed me to tweak hypo-parameters with no change to the code via the neat_config.txt file. This gave me the freedom to concentrate on experimentation. 
The built-in `StdOutReporter` function was very useful for displaying the results of each generation and the subsequent winning genome. I also found the [Pygame](https://github.com/pygame/pygame) 
library to be great for building quick and easy GUIs. Definitely one I'll use in future!

### How to:
- install `neat` and `pygame` libraries using
`pip install -r requirements.txt` 

- configure algorithm by editing neat_config.txt file

- run experiment using the command
`py main.py`
