# CIFAR & MNIST Joint Classification

### Requirements

Our implementation was tested in Python&nbsp;3.6.5.  Minimum testing was performed with&nbsp;3.7.1 but `requirements.txt` may need to change depending on your local Python configuration.  It uses the [PyTorch](https://pytorch.org/) neural network framework, version&nbsp;1.8.1.  Note that this version of PyTorch has issues with some GPU drivers.  We also performed minimal testing with version 1.7.1.  For the full requirements, see `requirements.txt` in the `src` directory.

We recommend running our program in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).  Once your virtual environment is created and *active*, run the following in the `src` directory:

```
pip install --user --upgrade pip
pip install -r requirements.txt
```

### Viewing Influence Rankings

We use the [Weights & Biases](https://wandb.ai/) package `wandb` to visualize ranking results.  W&B is free to [register](https://wandb.ai/login?signup=true) and use.  

You should enable your W&B account on your local system before running the program.  This is done by calling:

```
pip install wandb
wandb login
```

If you want to run without W&B, set the variable `USE_WANDB` in `poison/_config.py` to `False`.


### Running the Program

To run the program, simply call `./run.sh` in this directory.  All datasets are downloaded automatically and installed in the directory `.data`.  
