CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.0001 --batch-size 32 --n-epochs 30 --n-rotations 1 --n-cvruns 5 --cv-run 0 --norm --train --fp-save model_checkpoints/cnn_rgb/default/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.0001 --batch-size 32 --n-epochs 30 --n-rotations 1 --n-cvruns 5 --cv-run 1 --norm --train --fp-save model_checkpoints/cnn_rgb/default/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.0001 --batch-size 32 --n-epochs 30 --n-rotations 1 --n-cvruns 5 --cv-run 2 --norm --train --fp-save model_checkpoints/cnn_rgb/default/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.0001 --batch-size 32 --n-epochs 30 --n-rotations 1 --n-cvruns 5 --cv-run 3 --norm --train --fp-save model_checkpoints/cnn_rgb/default/ --fp-data data/plant_rgb/
CUDA_VISIBLE_DEVICES=0 python3 ../main_rgb.py --lr 0.0001 --batch-size 32 --n-epochs 30 --n-rotations 1 --n-cvruns 5 --cv-run 4 --norm --train --fp-save model_checkpoints/cnn_rgb/default/ --fp-data data/plant_rgb/
