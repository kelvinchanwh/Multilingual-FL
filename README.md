# Split Pretrained Models for Multilingual Federated Learning

## Enviroment Setup
`pip install -r requirements.txt`

## Data Setup
0. After deciding which data setup you would like, look for the corresponding dataset in `create_data` For the sake of this readme, we will use the `nc` data.
1. `cd` into the folder (`cd create_data/make_nc_data`)
2. Follow the instructions in the `readme` located in the folder. It will typically have scripts for downloading, preprocessing, splitting, and then moving the data into the final location for the model.

## Training/Evaluating Federated Learning Models
0. Make sure the enviroment and the data have been set up as above.
1. Depending on the type of model you want to train (classification, LM, or MT) see the corresponding scripts in `bin/run_fl_{mt,tc,lm}.sh`. Each script contains information about how to run centralized, non-IID FL, or IID FL learning, as well as random initialization and/or evaluation.

