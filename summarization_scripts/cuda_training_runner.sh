CUDA_VISIBLE_DEVICES=0 python training_runner.py 2>&1 | tee logs/train_$(date '+%Y-%m-%d').log
