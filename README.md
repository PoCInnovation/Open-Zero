# Open-Zero

## Table of Content

[Introduction](#introduction)  
[Features](#features)  
[Installation](#installation)  
[Quickstart](#quickstart)  
[Contributors](#contributors)  

------------
## Introduction

Open-Zero is a research project that aims to make open source implementation of [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) and [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules) methods from [DeepMind](https://github.com/deepmind) on the game of chess.

We use Deep Reinforcement Learning methods such as Asynchronous Advantage Actor-Crique or [A3C](https://paperswithcode.com/method/a3c).

![Schema](./.github/assets/muzero.png)

------------
## Features

### Training

The AI train by playing against itself using [A3C](https://paperswithcode.com/method/a3c) methods.

### Testing

We can test the AI multiple ways:
- Watch the AI play against itself
- Make an evaluation of a game using Stockfish

------------
## Installation

### Clone Repository
```
git clone https://github.com/PoCInnovation/Open-Zero.git
cd Open-Zero
```

### Install dependencies
```
pip3 install -r requirement.txt
```

------------
## Quickstart

The ```launch-project.sh``` script is the tool you use to do almost everything in this project.
Get the usage help by doing:
```
./launch-project.sh -h
```
------------
## Contributors

Gino Ambigaipalan → [Github](https://github.com/Tacos69)  
Jean-Baptiste Debize → [Github](https://github.com/jeanbaptistedebize)  
Nell Fauveau → [Github](https://github.com/Nellousan)  
Bogdan Guillemoles → [Github](https://github.com/bogdzn)  

------------
