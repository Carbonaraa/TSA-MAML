# README

## Dataset
For CIFAR-FS dataset please go this [Download link](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view)

For Tieredimagenet dataset please go this [Download link](https://drive.google.com/file/d/1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG/view)

For Toy dataset please go these 
[Download link (Aircraft)](https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view)
[Download link (Bird)](https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view)
[Download link (Car)](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
[Download link (Fungi)](https://drive.google.com/file/d/1IJk93N48X0rSL69nQ1Wr-49o8u0e75HM/view)
and partition the train and test set according to our appendix. In order to accelerate training phase, please collect this dataset into a pkl cache file using `import pickle`

## Environment

Tensorflow == 1.13

cuda == 10.0

cudnn == 7.6

python == 3.6.8

numpy == 1.18

opencv==3.4

pillow == 7.1

scikit-learn

tqdm



## Quick test via loading trained model
To quick evalute with our trained model, please download model from this [Download link (Cifar-100 models)](https://drive.google.com/file/d/13HFUTY4NTs2ZZpi5E8-eSBcWAce_ks4Q/view?usp=sharing). 

Test CIFAR-FS 5-way 1-shot MAML
	
	python train_maml.py --train=False --dataset cifar100 --meta_batch_size 4 --update_batch_size 1 \
		--update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w1s \
		--num_test_tasks 600 --num_filters 32 --max_pool True 

Test CIFAR-FS 5-way 1-shot TSA-MAML

	python train_tsamaml.py --train=False --dataset cifar100 --metatrain_iterations 40000 \
		--meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 \
		--logdir logs/tsa_maml_cifar5w1s --premaml logs/maml_cifar5w1s/bestmodel --num_test_tasks 600 \
		--num_filters 32 --max_pool True --num_groups 5

For other settings, you can quickly access the commands in ``train.sh`` and ``test.sh``.


## Train

At first, train a vallina MAML for task solution clustering.

**CIFAR100**
For N-way K-shot tasks (default group number is 5)

	python train_maml.py --train=False --dataset cifar100 --metatrain_iterations 60000 \
		--meta_batch_size 4 --update_batch_size <K> --update_lr 0.01 --num_updates 5 \
		--num_classes <N> --logdir <logdir> --num_test_tasks 600 --num_filters 32 \
		--max_pool True --num_groups 5

**TieredImageNet**
For N-way K-shot tasks (default group number is 5)

	python train_maml.py --train=False --dataset tiered --metatrain_iterations 100000 \
		--meta_batch_size 4 --update_batch_size <K> --update_lr 0.01 --num_updates 5 \
		--num_classes <N> --logdir <logdir> --num_test_tasks 600 --num_filters 32 \
		--max_pool True --num_groups 5


Then load this pretrained MAML and do training for TSA-MAML.

**CIFAR100**
For N-way K-shot tasks (default group number is 5)

	python train_tsamaml.py --train=False --dataset cifar100 --metatrain_iterations 40000 \
		--meta_batch_size 4 --update_batch_size <K> --update_lr 0.01 --num_updates 5 \
		--num_classes <N> --logdir <logdir> --premaml <ModelPathOfMAML> --num_test_tasks 600 \
		--num_filters 32 --max_pool True --num_groups 5 --cosann=True 

**TieredImageNet**
For N-way K-shot tasks (default group number is 5)

	python train_tsamaml.py --train=False --dataset tiered --metatrain_iterations 60000 \
		--meta_batch_size 4 --update_batch_size <K> --update_lr 0.01 --num_updates 5 \
		--num_classes <N> --logdir <logdir>  --premaml <ModelPathOfMAML> --num_test_tasks 600 \
		--num_filters 32 --max_pool True --num_groups 5 --cosann=True 

## Test
Same as the Quick test section.

## Citation

	@inproceedings{zhou2020task,
		title={Task Similarity Aware Meta Learning: Theory-inspired Improvement on MAML},
		author={Zhou, Pan and Zou, Yingtian and Yuan, Xiaotong and Feng, Jiashi and Xiong, Caiming and Hoi, SC},
		booktitle={4th Workshop on Meta-Learning at NeurIPS},
		year={2020}
	}
