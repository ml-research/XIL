# XIL
Repository for the Paper ``Right for the Wrong Scientific Reasons: Revising Deep Networks by Interacting with their 
Explanations'' by Patrick Schramowski, Wolfgang Stammer, Stefano Teso, Anna Brugger, Franziska Herbert, Xiaoting Shao, 
Hans-Georg Luigs, Anne-Katrin Mahlein & Kristian Kersting.

## Plant Disease Detection Datasets
TODO License \
Please get in touch with the corresponding authors.

## Revising of Scientific dataset
Default Training:
```
python3 classification/HyperspecConv/train.py 
    --data_path <path/to/data> 
    --save_path=<path> 
    --gpus=0,1,2,3 -b 10 --lr 0.0001 -j 5 --mask 0 --cv_splits 5 --cv_current_split 0
    --epochs=300
```
Revising:
```
python3 classification/HyperspecConv/train.py
    --data_path <path/to/data> 
    --save_path=<path> 
    --gpus=0,1,2,3 -b 10 --lr 0.0001 -j 5 --mask 2 --l2_grad=20 --cv_splits 5 --cv_current_split 0 
    --epochs=300 --weighted-rrr
```
Evaluating:
```
python3 classification/HyperspecConv/train.py
    --data_path <path/to/data> 
    --save_path=<path/tmp> 
    --resume=<path/model.pth.tar>
    --gpus=0,1,2,3
    --evaluate -b 10 -j 5 --mask=0 --cv_splits=5 --cv_current_split=0
```