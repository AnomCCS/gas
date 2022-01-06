# NLP Poison Adversarial Set and Target Identification

### Requirements

Our implementation was tested in Python&nbsp;3.6.5.  Minimum testing was performed with&nbsp;3.7.1 but `requirements.txt` may need to change depending on your local Python configuration.  It uses the [PyTorch](https://pytorch.org/) neural network framework, version&nbsp;1.8.1.  Note that this version of PyTorch has issues with some GPU drivers.  We also performed minimal testing with version 1.7.1.  For the full requirements, see `requirements.txt` in the `src` directory.

We recommend running our program in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).  Once your virtual environment is created and *active*, run the following in the `src` directory:

```
pip install --user --upgrade pip
pip install -r requirements.txt
```

### Running the Program

To run the program, simply call `scripts/run_sentiment.sh .` in this directory.  A representative dataset is present in the `.data` directory.
