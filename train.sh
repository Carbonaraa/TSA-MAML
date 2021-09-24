# train MAML
# CUDA_VISIBLE_DEVICES="1" python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w1s --num_filters 32 --max_pool=True 
# CUDA_VISIBLE_DEVICES="2" python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w5s --num_filters 32 --max_pool=True 
# CUDA_VISIBLE_DEVICES="3" python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_cifar10w1s --num_filters 32 --max_pool=True 
# CUDA_VISIBLE_DEVICES="4" python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_cifar10w5s --num_filters 32 --max_pool=True


# train TSA-MAML
# CUDA_VISIBLE_DEVICES="1" python train_tsamaml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir=logs/tsa_maml_cifar5w1s --premaml logs/maml_cifar5w1s/bestmodel  --num_filters 32 --max_pool=True --num_groups 5 --cosann=True --meta_lr=0.001
# CUDA_VISIBLE_DEVICES="2" python train_tsamaml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir=logs/tsa_maml_cifar5w5s --premaml logs/maml_cifar5w5s/bestmodel  --num_filters 32 --max_pool=True --num_groups 10 --cosann=True --meta_lr=0.001
# CUDA_VISIBLE_DEVICES="3" python train_tsamaml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir=logs/tsa_maml_cifar10w1s --premaml logs/maml_cifar10w1s/bestmodel  --num_filters 32 --max_pool=True --num_groups 5 --cosann=True --meta_lr=0.001
# CUDA_VISIBLE_DEVICES="4" python train_tsamaml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir=logs/tsa_maml_cifar10w5s --premaml logs/maml_cifar10w5s/bestmodel  --num_filters 32 --max_pool=True --num_groups 5 --cosann=True --meta_lr=0.001

