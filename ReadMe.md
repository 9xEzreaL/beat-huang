## WBC classification 
This projects mainly for saving kaiyu's life, try your best to beat SOTA then save her.

## Data Preparation
In "util/configuration.py" 
1. assign your dataset location
2. assign saving parameters location

## Training
```
CUDA_VISIBLE_DEVICES=0 python main.py --net my_densenet --experiment_name densenet121_SGD --bs_per_gpu 30
```

## Supported network
Choose your network add parser --net 
1. Desnsenet121 -> my_densenet
2. Resnext -> resnext
3. Resnest -> resnest

Add your oen model in "network/vision_mode.py"

Good luck!!!! beat the monster.
