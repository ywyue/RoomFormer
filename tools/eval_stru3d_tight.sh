#!/usr/bin/env bash

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=checkpoints/roomformer_stru3d_tight.pth \
               --output_dir=eval_stru3d_tight \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
