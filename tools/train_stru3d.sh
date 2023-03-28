#!/usr/bin/env bash

python main.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_stru3d
