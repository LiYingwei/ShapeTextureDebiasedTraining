### Training Scripts
Please set dataset path and output path by
```
DATA=/path/to/ImageNet
MODEL_DIR=/path/to/output
``` 
#### ResNet-50
```
python imagenet.py -a resnet50 --data $DATA --epochs 100 --schedule 30 60 90 --checkpoint $MODEL_DIR --gpu-id 0,1,2,3,4,5,6,7 --train-batch 512 --lr 0.2 --mixbn --style --multi_grid 
```
#### ResNet-101
```
python imagenet.py -a resnet101 --data $DATA --epochs 100 --schedule 30 60 90 --checkpoint $MODEL_DIR --gpu-id 0,1,2,3,4,5,6,7 --train-batch 256 --lr 0.1 --mixbn --style --multi_grid 
```
#### ResNet-152
```
python imagenet.py -a resnet152 --data $DATA --epochs 100 --schedule 30 60 90 --checkpoint $MODEL_DIR --gpu-id 0,1,2,3,4,5,6,7 --train-batch 256 --lr 0.1 --mixbn --style --multi_grid 
```
#### Mixup-ResNeXt-101
```
python imagenet.py -a resnext101 --data $DATA --epochs 180 --schedule 60 120 160 --checkpoint $MODEL_DIR --gpu-id 0,1,2,3,4,5,6,7 --train-batch 256 --lr 0.1 --mixbn --style --multi_grid --min_size 200 --mixup 0.4  
```
#### CutMix-ResNeXt-101
```
python imagenet.py -a resnext101 --data $DATA --epochs 210 --schedule 75 150 180 --checkpoint $MODEL_DIR --gpu-id 0,1,2,3,4,5,6,7 --train-batch 256 --lr 0.1 --mixbn --style --cutmix 1.0 
```
