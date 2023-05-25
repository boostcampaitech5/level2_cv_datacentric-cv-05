# --my_opt [Adam(default), SGD, RMSprop, Adagrad, AdamW ]
# --my_sched [MultiStepLR(default), CosineAnnealingLR, ReduceLROnPlateau]

# optimizer
python train.py --my_opt Adam --max_epoch 50 --wandb_name '1-1'
python train.py --my_opt Adam --my_sched CosineAnnealingLR --max_epoch 50 --wandb_name '1-2'
python train.py --my_opt Adam --my_sched ReduceLROnPlateau --max_epoch 50 --wandb_name '1-3'

python train.py --my_opt SGD --max_epoch 50 --wandb_name '2-1'
python train.py --my_opt SGD --my_sched CosineAnnealingLR --max_epoch 50 --wandb_name '2-2'
python train.py --my_opt SGD --my_sched ReduceLROnPlateau --max_epoch 50 --wandb_name '2-3'

python train.py --my_opt RMSprop --max_epoch 50 --wandb_name '3-1'
python train.py --my_opt RMSprop --my_sched CosineAnnealingLR --max_epoch 50 --wandb_name '3-2'
python train.py --my_opt RMSprop --my_sched ReduceLROnPlateau --max_epoch 50 --wandb_name '3-3'

python train.py --my_opt Adagrad --max_epoch 50 --wandb_name '4-1'
python train.py --my_opt Adagrad --my_sched CosineAnnealingLR --max_epoch 50 --wandb_name '4-2'
python train.py --my_opt Adagrad --my_sched ReduceLROnPlateau --max_epoch 50 --wandb_name '4-3'

python train.py --my_opt AdamW --max_epoch 50 --wandb_name '5-1'
python train.py --my_opt AdamW --my_sched CosineAnnealingLR --max_epoch 50 --wandb_name '5-2'
python train.py --my_opt AdamW --my_sched ReduceLROnPlateau --max_epoch 50 --wandb_name '5-3'
