# Softmax Federated Learning

## Directories
data: Store train/test/validation dataset  
logs: Save Tensorboard logs  
options: Save configuration JSON files  
results: Save test results in NPY format    
src: Source code

## Command
`python <path/to/project>/src/main.py <configuration JSON file name without directory or .json extension>`

e.g.) `python ./src/main.py case0`

## Options
+ epochs: Number of communication rounds
+ num\_users: Number of clients (or parties)
+ frac: Fraction of clients to use in global model update
+ local\_ep: Number of epochs in a single client (only for federated learning)
+ local\_bs: Batch size in a single client (only for federated learning)
+ optimizer: Optimizer type (sgd or adam)
+ lr: Learning rate
+ momentum: Momentum (only for SGD)
+ seed: Random seed
+ noise: Type of noise to give in training data (gaussian, saltandpepper, speckle)
+ noise\_frac: Fraction of clients to be trained with noised training data (federated learning) or fraction of training data to apply noise (non-federated learning)
+ federated: 1 for federated learning, 0 for non-federated learning
+ iid: 1 for IID(independent identical distribution), 0 for non-IID
+ clean: Number of clients to exclude before global model update (only for federated learning)
+ dist: Distance to use (euclidean) (only for federated learning)
+ gpu: GPU ID
