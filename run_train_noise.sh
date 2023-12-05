#Train mtl models with increasing noise parameter for subset of data 
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.2 --mtl
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.4 --mtl
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.6 --mtl
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.8 --mtl
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 1.0 --mtl

#Train vanilla PINN with increasing noise parameter for subset of data
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.2
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.4
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.6
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 0.8
python train.py --epochs 100 --batch 64 --lr 0.001 --noise 1.0