#!/usr/bin/env bash

python main.py --dataset_name=scenecad \
               --dataset_root=data/scenecad \
               --lr=5e-5 \
               --epochs=400 \
               --lr_drop=[320] \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_scenecad
