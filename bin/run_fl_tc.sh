#!/bin/sh

# centralized
python3 main_lm.py --data nc --model xlm-roberta-base --n_cpus 1 --n_gpus 1 --batch_size 8 --batch_accum 4 --lang_mix 0.99 --centralized --n_iterations 5 --lr 1e-5 --critical_layer 0 > nc_central_c0.txt

# IID FL
python3 main_lm.py --data nc --model xlm-roberta-base --n_cpus 1 --n_gpus 1 --batch_size 8 --batch_accum 4 --lang_mix 0.99 --n_iterations 5 --lr 1e-5 --critical_layer 0 > nc_iid_c0.txt

# Non-IID FL
python3 main_lm.py --data nc --model xlm-roberta-base --n_cpus 1 --n_gpus 1 --batch_size 8 --batch_accum 4 --lang_mix 0.0 --n_iterations 5 --lr 1e-5 --critical_layer 0 > nc_noniid_c0.txt

# For eval add "--n_iterations 0 --load_model <PATH_TO_MODEL.pt>"
# For random initialization add "--random_init" to the model and change n_iterations to 50.
