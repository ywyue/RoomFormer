#!/usr/bin/env bash

python main.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=2800 \
               --num_polys=70 \
               --semantic_classes=19 \
               --job_name=train_stru3d_sem_rich
