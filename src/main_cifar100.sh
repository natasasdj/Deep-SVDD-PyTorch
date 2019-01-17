#!/bin/bash
for j in `seq 1 10`;
do
for i in `seq 0 19`;
do
echo $j, $i
#python3 main.py cifar100 cifar10_LeNet ../log/cifar100_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class $i
done
done
