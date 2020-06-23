# XIL
Repository for the Paper ``Making deep neural networks right for the right scientific reasons by interacting with their explanations'' by Patrick Schramowski, Wolfgang Stammer, Stefano Teso, Anna Brugger, Franziska Herbert, Xiaoting Shao, 
Hans-Georg Luigs, Anne-Katrin Mahlein & Kristian Kersting.

In the process of paper review we have generated a Code Ocean capsule found under: TODO.

## Plant Disease Detection Datasets
TODO License \
Please get in touch with the corresponding authors.

## Revising of Scientific dataset
Default Training:
```
python3 Plant_Phenotyping/main_hs.py
    --data_path <path/to/data> 
    --save_path=<path> 
    --gpus=0,1,2,3 -b 10 --lr 0.0001 -j 5 --mask 0 --cv_splits 5 --cv_current_split 0
    --epochs=300
```
Revising:
```
python3 Plant_Phenotyping/main_hs.py
    --data_path <path/to/data> 
    --save_path=<path> 
    --gpus=0,1,2,3 -b 10 --lr 0.0001 -j 5 --mask 2 --l2_grad=20 --cv_splits 5 --cv_current_split 0 
    --epochs=300 --weighted-rrr
```
Evaluating:
```
python3 Plant_Phenotyping/main_hs.py
    --data_path <path/to/data> 
    --save_path=<path/tmp> 
    --resume=<path/model.pth.tar>
    --gpus=0
    --evaluate -b 4 -j 5 --mask=0 --cv_splits=5 --cv_current_split=0
```
Creating Explanations:
```
python3 Plant_Phenotyping/main_hs.py
    --data_path <path/to/data> 
    --save_path=<path/tmp> 
    --resume=<path/model.pth.tar>
    --gpus=0
    --gradcam -b 1 --mask=0 --cv_splits=5 --cv_current_split=0
```

## TODO:
- Add example RRR on DecoyMNIST
