CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 60 --n-rotations 1 --n-cvruns 5 --cv-run 0 --norm --rrr --l2-grads 1 --train --fp-save model_checkpoints/cnn_rgb/rrr/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 60 --n-rotations 1 --n-cvruns 5 --cv-run 1 --norm --rrr --l2-grads 1 --train --fp-save model_checkpoints/cnn_rgb/rrr/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 60 --n-rotations 1 --n-cvruns 5 --cv-run 2 --norm --rrr --l2-grads 1 --train --fp-save model_checkpoints/cnn_rgb/rrr/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 60 --n-rotations 1 --n-cvruns 5 --cv-run 3 --norm --rrr --l2-grads 1 --train --fp-save model_checkpoints/cnn_rgb/rrr/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.00005 --batch-size 32 --n-epochs 60 --n-rotations 1 --n-cvruns 5 --cv-run 4 --norm --rrr --l2-grads 1 --train --fp-save model_checkpoints/cnn_rgb/rrr/ --fp-data data/plant_rgb/
