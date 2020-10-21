# BoxE
This repository contains the source code for the BoxE model, presented at NeurIPS 2020 in the [paper](https://arxiv.org/pdf/2007.06267.pdf) "BoxE: A Box Embedding Model for Knowledge Base Completion". The repository includes all evaluation datasets, code for training and testing BoxE on these datasets to reproduce results presented in the paper, and a D3.JS visualization tool, BoxEViz, to show the evolution of box embeddings over the course of training. 

## Requirements
- TensorFlow 1.x (>=1.15.0) and its corresponding NumPy version
- msgpack 0.6.1 
- msgpack-numpy 0.4.4.2 for dataset loading

## Datasets
The repository includes the FB15-237, WN18RR and YAGO3-10 knowledge graph completion datasets, the JF17K and FB-AUTO higher-arity knowledge base benchmarks, as well as SportsNELL, the dataset created for the evaluation of rule injection with BoxE.

## Running BoxE
Training BoxE is primarily done through the Training.py file, and evaluation through the Testing.py file. Command-line arguments for these files are explained in detail within their respective help descriptions, accessible via the -h flag following the Python file call, that is, python Training.py -h (respectively, python Testing.py -h)

## Reproducing Results
We provide the set of commands per dataset with the optimal hyper-parameters for a simple reproduction of results reported in the paper.

#### FB15k-237 - Uniform Negative Sampling:
Train: ```python Training.py "FB15k-237" -epochs 1000 -validCkpt 100 -learningRate 0.0001 -nbNegExp 100 -lossMargin 12 -embDim 500 -batchSize 1024 -negSampling unif  -lossOrd 1 -useTB F``` 

Test: ```python Testing.py "FB15k-237" rank -embDim 500 -lossOrd 1```

#### FB15k-237 - Self-Adversarial Negative Sampling:
Train: ```python Training.py "FB15k-237" -epochs 1000 -validCkpt 100 -learningRate 0.00005 -nbNegExp 100 -lossMargin 3 -embDim 1000  -batchSize 1024 -negSampling selfadv -advTemp 4.0 -useTB F```

Test:  ```python Testing.py "FB15k-237" rank -embDim 1000```

#### WN18RR - Uniform:
Train: ```python Training.py "WN18RR" -epochs 1000 -validCkpt 100 -learningRate 0.001 -nbNegExp 150 -lossMargin 5 -embDim 500 -batchSize 512 -negSampling unif -useTB F```

Test:  ```python Testing.py "WN18RR" rank -embDim 500```

#### WN18RR - Self-Adversarial
Train: ```python Training.py "WN18RR" -epochs 1000 -validCkpt 100 -learningRate 0.001 -nbNegExp 150 -lossMargin 3 -embDim 500 -batchSize 512 -negSampling selfadv -advTemp 2.0 -useTB F```

Test:  ```python Testing.py "WN18RR" rank -embDim 500```

#### YAGO3-10 - Uniform:
Train: ```python Training.py "YAGO3-10" -epochs 400 -validCkpt 100 -learningRate 0.001 -nbNegExp 150 -lossMargin 10.5 -embDim 200 -batchSize 4096 -negSampling unif -useTB F -augmentInv T```

Test:  ```python Testing.py "YAGO3-10" rank -embDim 200 -augmentInv T```

#### YAGO3-10 - Self-Adversarial:
Train: ```python Training.py "YAGO3-10" -epochs 400 -validCkpt 100 -learningRate 0.001 -nbNegExp 100 -lossMargin 6 -embDim 200 -batchSize 4096 -negSampling selfadv -advTemp 2.0 -useTB F -augmentInv T```

Test:  ```python Testing.py "YAGO3-10" rank -embDim 200 -augmentInv T```

#### JF17K - Uniform:
Train: ```python Training.py "JF17K" -epochs 1000 -validCkpt 100 -learningRate 0.002 -nbNegExp 100 -lossMargin 15 -embDim 200 -batchSize 1024 -negSampling unif -useTB F```

Test:  ```python Testing.py "JF17K" rank -embDim 200```

#### JF17K - Self-Adversarial:
Train: ```python Training.py "JF17K" -epochs 1000 -validCkpt 100 -learningRate 0.0001 -nbNegExp 100 -lossMargin 5 -embDim 200 -batchSize 1024 -negSampling selfadv -advTemp 2.0 -useTB F```

Test:  ```python Testing.py "JF17K" rank -embDim 200```

#### FB-AUTO - Uniform:
Train: ```python Training.py "FB-AUTO" -epochs 1000 -validCkpt 100 -learningRate 0.002 -nbNegExp 100 -lossMargin 18 -embDim 200 -batchSize 1024 -negSampling unif -useTB F```

Test:  ```python Testing.py "FB-AUTO" rank -embDim 200```

#### FB-AUTO - Self-Adversarial:
Train: ```python Training.py "FB-AUTO" -epochs 1000 -validCkpt 100 -learningRate 0.0005 -nbNegExp 100 -lossMargin 9 -embDim 200 -batchSize 1024 -negSampling selfadv -advTemp 2.0 -useTB F```

Test:  ```python Testing.py "FB-AUTO" rank -embDim 200```

#### SportsNELL - No Rule Injection:
Train: ```python Training.py "NELLRuleInjSplit90Mat" -epochs 2000 -validCkpt 200 -learningRate 0.001 -nbNegExp 100 -lossMargin 6 -embDim 200 -batchSize 1024 -negSampling unif -useTB F```

Test:  ```python Testing.py "NELLRuleInjSplit90Mat" rank -embDim 200```

Filtered Test: ```python Testing.py "NELLRuleInjSplit90Mat" rank -testFile test_subset.kbb -embDim 200```

#### SportsNELL - Rule Injection (BoxE+RI):
Train: ```python Training.py "NELLRuleInjSplit90Mat" -epochs 2000 -validCkpt 200 -learningRate 0.001 -nbNegExp 100 -lossMargin 6 -embDim 200 -batchSize 1024 -negSampling unif -useTB F -ruleDir RulesNELL.txt```

Test:  ```python Testing.py "NELLRuleInjSplit90Mat" rank -embDim 200 -ruleDir RulesNELL.txt```

Filtered Test: ```python Testing.py "NELLRuleInjSplit90Mat" rank -testFile test_subset.kbb -embDim 200 -ruleDir RulesNELL.txt```

## Visualization Extensions:

This BoxE implementation additionally supports 2 visualization options, namely:
- TensorBoard for observing the evolution of model loss, and 
- BoxEViz, a D3.JS tool for observing boxes and points over training.

### Using TensorBoard: 
To use the TensorBoard extension, simply set the flag -useTB T during training. Then, run Tensorboard via the command ```tensorboard --logdir summaries```, and open the resulting local server.

### Using BoxEViz: 
BoxEViz is a D3.JS tool that enables a visualization of the evolution of points and boxes over training.

#### Setting up BoxEViz:
The main settings of BoxEViz are stored in BoxEViz/settings.json. There, users can change the two visualized box dimensions (default, the first two dimensions), as well as the observed entities (default, the first six entities and their bumps). This can be done by simply setting different integer arrays at the "selectedDims" and "selectedEntities" entries. This should be done *before* running a training configuration so that the changes take effect. 
#### Running BoxEViz: 
Following a BoxE training run with the flag -viz T, a data.json file is generated in the BoxEViz directory. To visualize the data, simply run BoxEViz/main.html on a local server (otherwise the tool will not run. To start a server in Python 3, run ```python3 -m http.server```)
Once open, BoxEViz enables you to scroll through the different box configurations at every epoch, and select/deselect relations and entities to show.

##  Citing this paper
If you make use of this code, or its accompanying [paper](https://arxiv.org/pdf/2007.06267), please cite this work as follows:

```
@inproceedings{ACLS-NeurIPS2020,
  title={BoxE: A Box Embedding Model for Knowledge Base Completion},
  author    = {Ralph Abboud and
                {\.I}smail {\.I}lkan Ceylan and
               Thomas Lukasiewicz and Tommaso Salvatori},
  booktitle={Proceedings of the Thirty-Fourth Annual Conference on Advances in Neural Information Processing Systems ({NeurIPS})},
  year={2020}
}
```

