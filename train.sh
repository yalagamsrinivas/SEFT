#!/bin/bash
echo ">>>>>>>>>>>>> Train model $1 from scratch <<<<<<<<<<<<<<<<<<<"
python train_main.py \
--paths_text 'text/tweet' \
--paths_technical 'technical' \
--model_name $1 \