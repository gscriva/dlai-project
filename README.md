# A Scalable Model for Predicting Ground State Energy

## Problem 
Predict the lowest energy level of a one-dimensional ultracold atom subject to an external optical speckle field (disorder).
Our aim is to solve the Schrödinger equation and thus find the the ground state energy. 


## Goal
We want to reproduce the results achieved in [Scientific reports 9.1 (2019): 1-12.](https://www.nature.com/articles/s41598-019-42125-w), but using a scalable version of the network, as did in [Physical Review E 102.3 (2020): 033301](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.033301).


## Usage

### Requirements
Install all the requirements using `pip install -r requirements.txt`, we recommend to create an isolated environment before.

### Data structure
The input data must be organized as follow:
```
data
  ├── train_data_L14.npz
  ├── test_data_L14.npz
  ├── train_data_L28.npz
  ├── test_dataL_28.npz
  └── ...
```

They are not present in the git repo, but they are available in my own Drive.

## Train
To train with the simplest MLP model one has to run

```python main.py --train --data_dir data/train_dataL14.npz --input_size 15```.

Several commands are avialable to switch model, change MLP hidden structure, etc. Just run `python main.py --help`.
