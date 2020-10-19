# Interpretable Probabilistic Password Strength Meters via Deep Learning

From the [paper](https://arxiv.org/pdf/2004.07179.pdf).

<p align="center">
	<img src ="head.png" />
</p>

The notebook *Meter_interface_poc.ipynb*  contains a functional proof-of-concept for the interface of the meter. 

Requirements: 
* python3
* Tensorflow < 2.0
---

# Training:
Requirements 
* python3
* Tensorflow >= 2.0

You can train your model using the script *train.py*. This takes as input a configuration file (GIN configuration file).  This defines the architecture of the network and the training process.<br>

An example of configuration file can be found in *CONFs/ORIGINAL.gin*.  To note, every parameter in the file is mandatory.<br>

> py train.py CONFs/ORIGINAL.gin

In order to train on your password set, you have to modify *setup.home_train* by inserting the directory where the training set is located. In this directory you have to put two files:

1. A textual file **called *X.txt*** containing the training-set. The training-set is composed of a list of passwords (one per line) which **frequency is preserved**. For instance, if the password *"12345"* appears 10 times in the leak, this must appear 10 times in *X.txt*. **The set of passwords must be shuffled before the training.** 
2. A pickle file mapping chars to integers called ***charmap.pickle***. An example file can be found in *./charmap.pickle*. You can use the default one or create your own.


The training process continues till an early-stopping criteria based on the test-sets is reached. Although a maximum number of epochs can be expressed in the configuration file.

During the training, logs are saved inside *HOME/LOGs*, and can be visualized with *tensorbard*. At the end of the training, the model is saved as a keras model inside the directory */HOME/MODELs* 

# Pre-trained models:

*PRETRAINED_MODELs/*

Work in progress....

---
# Model Evaluation:

Work in progress....

---

How to cite our work:

> @InProceedings{10.1007/978-3-030-58951-6_25,<br>
> 	author="Pasquini, Dario
> 	and Ateniese, Giuseppe
> 	and Bernaschi, Massimo",<br>
> 	editor="Chen, Liqun
> 	and Li, Ninghui
> 	and Liang, Kaitai
> 	and Schneider, Steve",<br>
> 	title="Interpretable Probabilistic Password Strength Meters via Deep Learning",<br>
> 	booktitle="Computer Security -- ESORICS 2020",<br>
> 	year="2020",<br>
> 	publisher="Springer International Publishing",<br>
> 	address="Cham",<br>
> 	pages="502--522",<br>
> 	isbn="978-3-030-58951-6"
> }

