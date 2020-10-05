### Training Scripts
Please set dataset path and checkpoint path by
```
DATA=/path/to/ImageNet
CKPT=/path/to/checkpoint.pth.tar
``` 
#### ResNet-50
```
python imagenet.py -a resnet50 --data $DATA --mixbn --evaluate --load $CKPT 
```
#### ResNet-101
```
python imagenet.py -a resnet101 --data $DATA --mixbn --evaluate --load $CKPT
```
#### ResNet-152
```
python imagenet.py -a resnet152 --data $DATA --mixbn --evaluate --load $CKPT
```
#### Mixup-ResNeXt-101
```
python imagenet.py -a resnext101 --data $DATA --mixbn --evaluate --load $CKPT
```
#### CutMix-ResNeXt-101
```
python imagenet.py -a resnext101 --data $DATA --mixbn --evaluate --load $CKPT 
```
