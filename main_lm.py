import os
import random
import argparse
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
import traceback
import tqdm
import json
import datetime
import numbers
import os
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import *

import math
import bisect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common.typing import Scalar
import ray
from sacrebleu.metrics import BLEU, CHRF, TER

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    MT5Model,
    T5Tokenizer,
    MBartForCausalLM,
    DistilBertForSequenceClassification,
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    BartForCausalLM,
    logging,
    DistilBertForMaskedLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    set_seed
)

seed_val = 1
set_seed(seed_val)
print("Seed is", seed_val)
bleu = BLEU()

import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error() # hides warning about CasualLM not loading encoder

from dataset_utils import (
    LineByLineTextDataset,
    get_dataset,
    get_random_id_splits,
    make_collate_fn_wlang,
    make_collate_fn,
    get_dataloader,
    do_fl_partitioning_brown,
    do_fl_partitioning,
    MultilingualDataset,
    get_dataset_type,
    MTDataset,
)

from constants import *

BIG_FILE_CACHE = os.path.join(os.getcwd(), "cache")

## Global Vars that are set under `if __name__ == "__main__"`
ACCUM_STEPS = 1
BATCH_SIZE = 2
CUDA_COUNT = 0 # need to keep track for clients, iterative take the next one
RANDOM_INIT = False
MODEL_NAME = ""
DATA = ""
client_resources = {"num_gpus": 0, "num_cpus": 1} # NOTE: can do fractional GPUs, this is per process/client
GPU_MAPPING = {}
EVAL_NUM = 0
NUM_SKIP_EVAL = 1
PREV_LOSS = -1.0
PREV_ACC = 0.001
tokenizer = None
LEARNING_RATE = None
LANG_MIX = None
CACHE_STR = None
ALL_OPTIMIZERS = {}
TOP_N_SCORES = []
GLOBAL_LANG_MAP = None
C = 0
START_ROUND = 0
SKIP_DATASET_GEN = None

def save_history(hist, name):
  time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
  with open( f"{BIG_FILE_CACHE}/hist_{CACHE_STR}_{time}_{name}.json", "w", encoding="utf-8") as f:
            json.dump(hist.__dict__, f, ensure_ascii=False, indent=4)

# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, optimizer, device: str, cid: str = "", get_accuracy: bool = False):
    """Train the network on the training set."""
    global ACCUM_STEPS
    net.train()
    net.zero_grad()
    losses = []
    total, correct = 0, 0
    for _ in range(epochs):
        for batch_idx, batch in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
            input_ids = batch["input_ids"]
            label_ids = batch["labels"]
            attn_mask = batch["attention_mask"]

            label_ids = label_ids.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            x = {"input_ids": input_ids, "labels": label_ids, "attention_mask": attn_mask}
            output = net(**x)
            loss = output.loss / ACCUM_STEPS
            loss.backward() 
            losses.append(output.loss.cpu().detach().repeat(len(batch)))

            label_ids = label_ids.cpu()
            label_ids = label_ids.cpu()
            attn_mask = attn_mask.cpu()

            # if get_accuracy:
            pred_labels = output.logits.argmax(dim=-1).cpu()
            truth_labels = label_ids.squeeze(-1).cpu()
            correct += torch.sum(torch.eq(pred_labels, truth_labels)).cpu().detach().item()
            total += len(pred_labels)

            if (batch_idx + 1) % ACCUM_STEPS == 0:  
                optimizer.step()
                net.zero_grad()
                loss = 0

    net = net.cpu()
    loss = 0
    net.zero_grad()
    label_ids = label_ids.to("cpu")
    label_ids = label_ids.to("cpu")
    attn_mask = attn_mask.to("cpu")
    mean_loss = torch.cat(losses).mean()
    # if get_accuracy:
    print(f"TRAIN Accuracy for is {correct/total}")
    print(f"Got a TRAIN PPL value of {mean_loss.detach().item()} and {torch.exp(mean_loss).detach().item()} \
            for cid={cid}, label={batch['langs'][0].item()}")
    return mean_loss.detach().item()

def test(net, testloader, device: str, get_accuracy: bool = False):
    """Validate the network on the entire test set."""
    net.eval()
    losses = []
    correct = 0
    total = 0
    labels_to_losses = defaultdict(list)
    labels_to_accuracies = defaultdict(dict)
    with torch.no_grad():
        for idx, batch in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            input_ids = batch["input_ids"]
            label_ids = batch["labels"]
            attn_mask = batch["attention_mask"]
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            label_ids = label_ids.to(device)
            x = {"input_ids": input_ids, "labels": label_ids, "attention_mask": attn_mask}
            output = net(**x)
            # if get_accuracy:
            pred_labels = output.logits.argmax(dim=-1).cpu()
            truth_labels = label_ids.squeeze(-1).cpu()
            correct += torch.sum(torch.eq(pred_labels, truth_labels)).item()
            total += len(pred_labels)
            loss = output.loss
            # try:
            # assert len(set(batch["langs"].numpy().tolist())) == 1, set(batch["langs"].numpy().tolist())
            # labels_to_losses[batch["langs"][0].item()].append(output.loss.item())
            if "correct" not in labels_to_accuracies[batch["langs"][0].item()]:
                labels_to_accuracies[batch["langs"][0].item()]["correct"] = 0
                labels_to_accuracies[batch["langs"][0].item()]["total"] = 0
            labels_to_accuracies[batch["langs"][0].item()]["correct"] += torch.sum(torch.eq(pred_labels, truth_labels)).item()
            labels_to_accuracies[batch["langs"][0].item()]["total"] += len(pred_labels)
            # except Exception as e:
            #     print(f"Cant make lang ppls unless entire batch is the same: use a round num for batch size: {e}")
            losses.append(loss.repeat(len(batch)))

    # from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py
    mean_loss = torch.cat(losses).mean()
    max_num = max(list(labels_to_accuracies.keys()))
    accuracy_list = list()
    for label in range(max_num+1):
        # print(f"For EVAL Label {label} the average PPL is {np.exp(np.mean(labels_to_losses[label]))}")
        try:
            label_acc = labels_to_accuracies[label]['correct'] / labels_to_accuracies[label]['total']
            accuracy_list.append(label_acc)
            print(f"For EVAL Label {label} the average accuracy is {label_acc}")
        except KeyError:
            print (f"Label {label} not present in eval dataset")
    net = net.to("cpu")
    # if get_accuracy:
    print(f"EVAL Accuracy is {correct/total}")
    label_ids = label_ids.to("cpu")
    label_ids = label_ids.to("cpu")
    attn_mask = attn_mask.to("cpu")
    return mean_loss.item(), correct/total, accuracy_list

# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class RayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, optimizer, net):
        global CUDA_COUNT
        global GPU_MAPPING 
        global LEARNING_RATE

        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # instantiate model
        self.net = net
        self.optimizer = optimizer

        # determine device
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        if cuda_available:
            device_str = f"cuda:0" 
        elif mps_available:
            device_str = f"mps"
        else:
            device_str = f"cpu" # CUDA zero defaults to CUDA_VISIBLE_DEVICES
        self.device = torch.device(device_str)
        
    def get_local_params(self, prev_global_round):
        load_path = f"{BIG_FILE_CACHE}/{MODEL_NAME.split('/')[-1]}/{CACHE_STR}/"
        file_path = os.path.join(load_path, f"round-{prev_global_round}-{self.cid}-weights.npz")
        # Save nd_arrays
        print(f"Loading round {prev_global_round} ndarrays for cid {self.cid} from {file_path}")
        try:
            local_parameters = list(np.load(file_path).values())
        except FileNotFoundError:
            print (f"{file_path} not found")
            return None
        return local_parameters
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, global_parameters, local_parameters=None, c=0):
        if local_parameters is None:
            params_dict = zip(self.net.state_dict().keys(), global_parameters)
            state_dict = OrderedDict(
                {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
            )
        else:
            local_keys = list()
            for key in self.net.state_dict().keys():
                if "embeddings" in key:
                    local_keys.append(key)
                else:
                    try:
                        if int(key.split(".")[3]) >= int(c):
                            local_keys.append(key)
                    except IndexError:
                        pass
            params_dict = zip(self.net.state_dict().keys(), global_parameters, local_parameters)
            state_dict = OrderedDict()
            for k, v_global, v_local in params_dict:
                if k in local_keys:
                    state_dict[k] = torch.from_numpy(np.copy(v_local))
                else:
                    state_dict[k] = torch.from_numpy(np.copy(v_global))

        self.net.load_state_dict(state_dict, strict=True)
        

    def fit(self, parameters, config):
        global DATA
        global BATCH_SIZE
        c = config["c"]

        global tokenizer
        global_round = int(config["epoch_global"])

        if global_round < int(config["start_round"]):
            params = self.get_local_params(global_round)
            train_loss = 0
            data_length = 1600
        else:
            local_parameters = self.get_local_params(global_round-1)
            if local_parameters is None:
                local_parameters = parameters # Fallback to global parameters if local parameters does not exist

            self.set_parameters(parameters, local_parameters, c = c)

            try:
                # load data for this client and get trainloader
                num_workers = len(ray.worker.get_resource_ids()["CPU"])
                trainloader = get_dataloader(
                    self.fed_dir,
                    self.cid,
                    is_train=True,
                    batch_size=BATCH_SIZE,
                    workers=num_workers,
                    data=DATA,
                    tokenizer=tokenizer,
                    shuffle=True,
                    lang_mix=LANG_MIX,
                )

                # send model to device
                self.net.to(self.device)

                # train
                train_loss = train(self.net, trainloader, int(config["epochs"]), self.optimizer, device=self.device, 
                        cid=self.cid, get_accuracy=("pawsx" in DATA or "nc" in DATA))
            
                params = self.get_parameters(self.device)

                data_length = len(trainloader.dataset)

            except Exception as e:
                print(f"Error failed in train was `{e}`")
                print(traceback.format_exc())
                raise e
    

        if params is not None and global_round >= int(config["start_round"]):
            save_path = f"{BIG_FILE_CACHE}/{MODEL_NAME.split('/')[-1]}/{CACHE_STR}/"
            file_path = os.path.join(save_path, f"round-{global_round}-{self.cid}-weights.npz")
            # Save nd_arrays
            print(f"Saving round {global_round} ndarrays for cid {self.cid} to {file_path}")
            np.savez(file_path, *params)

        # return local model and statistics
        return params, data_length, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        global tokenizer
        
        c = config["c"]
        global_round = int(config["epoch_global"])

        self.set_parameters(parameters, None, c)

        print(f"[Client {self.cid}] evaluate, config: {config}")
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=BATCH_SIZE, workers=num_workers, data=DATA,
            tokenizer=tokenizer, shuffle=False, lang_mix=LANG_MIX
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy, accuracy_list = test(self.net, valloader, device=self.device)
        self.net.to("cpu")
        # return statistics
        return float(loss), len(valloader.dataset), {f"local_accuracy_{self.cid}": float(accuracy), f"local_accuracy_{self.cid}_label": accuracy_list}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    global BATCH_SIZE
    global C
    global START_ROUND
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(BATCH_SIZE),
        "c": str(C),
        "start_round": str(START_ROUND)
    }
    return config
    
def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size"""
    global BATCH_SIZE
    global C
    config = {
        "epoch_global": str(rnd),
        "batch_size": str(BATCH_SIZE),
        "c": str(C)
    }
    return config

def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """

    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    output =  {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }
    return output

class TopItem:
    # to manage saving the top_N items
    def __init__(self, score: float, path: str):
        self.score = score
        self.path = path
    
    def __lt__(self, other) -> bool:
        return self.score < other.score

    def to_str(self) -> str:
        return f"Score: {self.score} at Path: {self.path}"


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArray) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset, lang_mix: float
) -> Callable[[fl.common.NDArray], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, weights: fl.common.NDArray, fed_eval_config) -> Optional[Tuple[float, float]]:
        """Use the entire test set for evaluation."""
        global CUDA_COUNT
        global BATCH_SIZE
        global DATA
        global GPU_MAPPING
        global EVAL_NUM
        global MODEL_NAME
        global tokenizer
        global PREV_LOSS
        global PREV_ACC
        global NUM_SKIP_EVAL
        global TOP_N_SCORES
        KEEP_PILE = 2

        # if EVAL_NUM % NUM_SKIP_EVAL == 1: # after every epoch basically
        #     print(f"Skipping with EVAL_NUM={EVAL_NUM} and NUM_SKIP_EVAL={NUM_SKIP_EVAL}")
        #     EVAL_NUM += 1
        #     return PREV_LOSS, {"perplexity": math.exp(PREV_LOSS), "accuracy": PREV_ACC}

        model = make_huggingface_model()
        
        set_weights(model, weights)

        # determine device
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        if cuda_available:
            device_str = f"cuda:0" 
        elif mps_available:
            device_str = f"mps"
        else:
            device_str = f"cpu" # CUDA zero defaults to CUDA_VISIBLE_DEVICES

        device = torch.device(device_str)
        model.to(device)

        batch_size = BATCH_SIZE                

        testloader = get_dataloader(
            testset, -1, is_train=False, batch_size=batch_size, workers=3, 
            tokenizer=tokenizer, shuffle=False, data=DATA, lang_mix=LANG_MIX
        )
        
        loss, accuracy, accuracy_list = test(model, testloader, device=device, get_accuracy=("pawsx" in DATA or "nc" in DATA))
        
        if len(TOP_N_SCORES) < KEEP_PILE or loss < TOP_N_SCORES[-1].score:
            if "_cont" in CACHE_STR:
                save_path = f"{BIG_FILE_CACHE}/{MODEL_NAME.split('/')[-1][-3]}/{CACHE_STR}/"
            else:
                save_path = f"{BIG_FILE_CACHE}/{MODEL_NAME.split('/')[-1]}/{CACHE_STR}/"

            bisect.insort(TOP_N_SCORES, TopItem(loss, save_path + f"{EVAL_NUM}.pt"))
            print([item.to_str() for item in TOP_N_SCORES])
            TOP_N_SCORES, to_remove = TOP_N_SCORES[:KEEP_PILE], TOP_N_SCORES[KEEP_PILE:]

            for top_item in to_remove:
                print(f"Removing {top_item.to_str()}")
                os.remove(top_item.path)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

        
            torch.save(model, f"{save_path}/{EVAL_NUM}.pt")


        EVAL_NUM += 1

        PREV_LOSS = loss
        PREV_ACC = accuracy

        # return statistics
        return loss, {"accuracy": accuracy, "accuracy_per_label": accuracy_list}

    return evaluate


def make_tokenizer(model_name: str):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return tokenizer


def make_huggingface_model():
    global CACHE_STR
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
    except Exception as e:
        pass # loading model doesn't need this

    warnings.filterwarnings("ignore")
    logging.set_verbosity_error() # hides warning about CasualLM not loading encoder
    if ".pt" in MODEL_NAME[-3:]:
        print(f"Loading model {MODEL_NAME}")
        model = torch.load(MODEL_NAME)
        CACHE_STR = MODEL_NAME.split("/")[-2] + "_cont"
    elif not RANDOM_INIT:
        if "gpt2" in MODEL_NAME:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                from_tf=False,
                cache_dir=f"{BIG_FILE_CACHE}/huggingface_cache/"
            )
        elif "xlm" in MODEL_NAME:
            try:
                model = XLMRobertaForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    cache_dir=f"{BIG_FILE_CACHE}/huggingface_cache/",
                    num_labels=10
                )
            except Exception as e:
                breakpoint()
                print(e)
        elif "bert" in MODEL_NAME:
            model = DistilBertForMaskedLM.from_pretrained(
                MODEL_NAME,
                cache_dir=f"{BIG_FILE_CACHE}/huggingface_cache/"
            )
        elif "m2m" in MODEL_NAME:
            model = M2M100ForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                cache_dir=f"{BIG_FILE_CACHE}/huggingface_cache/"
            )
        else:
            raise NotImplementedError(f"Haven't impleneted model={MODEL_NAME}")
            
    else:
        print("Training new model from scratch")
        if "xlm" in MODEL_NAME:
            config.num_labels = 10
            model = XLMRobertaForSequenceClassification(
                config
            )
        elif "bert" in MODEL_NAME:
            model = DistilBertForMaskedLM(config)
        elif "m2m" in MODEL_NAME:
            model = M2M100ForConditionalGeneration(config)

    return model

# Based on the original implementation by Vasundhara Agarwal
class CustomFedAvgStrategy(fl.server.strategy.FedAvg):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, fl.common.NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = False,
        initial_parameters: Optional[fl.common.Parameters] = None,
        fit_metrics_aggregation_fn: Optional[fl.common.MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[fl.common.MetricsAggregationFn] = None
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
       
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            save_path = f"{BIG_FILE_CACHE}/{MODEL_NAME.split('/')[-1]}/{CACHE_STR}/"
            # Convert `Parameters` to `List[np.ndarray]`
            ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_params)
            
            file_path = os.path.join(save_path, f"round-{server_round}-global-weights.npz")
            # Save nd_arrays
            print(f"Saving round {server_round} ndarrays for global model to {file_path}")
            np.savez(file_path, *ndarrays)
        return aggregated_params, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}
        loss_aggregated = weighted_loss_avg(
           [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ] 
        )
        metrics_aggregated = aggregate_weighted_average(
           [
                (evaluate_res.num_examples, evaluate_res.metrics)
                for _, evaluate_res in results
            ] 
        )
        return loss_aggregated, metrics_aggregated


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Prepares the data
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 4. Starts a Ray-based simulation where a % of clients are sample each round.
# 5. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="dataset path to use", required=True)
    parser.add_argument("--model", type=str, help="The model name to use", required=True)
    parser.add_argument("--n_cpus", type=int, help="The number of CPUs to use PER MACHINE", default=-1)
    parser.add_argument("--n_gpus", type=float, help="The number of GPUs to use TOTAL", default=0.0)
    parser.add_argument("--frac_fit", type=float, help="The percent of nodes to sample each time", default=1.0)
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use")
    parser.add_argument("--batch_accum", type=int, default=8, help="The batch accumulation steps to do")
    parser.add_argument("--n_iterations", type=int, default=120, help="The number of iterations to do")
    parser.add_argument("--lang_mix", type=float, default=0.0, help="The lang mixture to use (0 for separate, 1 for uniform)")
    parser.add_argument("--lr", type=float, default=5e-5, help="The learning rate to use for the optimizer")
    parser.add_argument("--random_init", action="store_true", help="Whether to load a random intialized model")
    parser.add_argument("--load_model", type=str, help="whether to load a saved model path")
    parser.add_argument("--centralized", action="store_true", help="Whether to run with a centralized run instead")
    parser.add_argument("--test", type=int, default=0, help="Whether to load small data instead")
    parser.add_argument("--critical_layer", type=int, default=0, help="Critical layer to split BERT")
    parser.add_argument("--start_round", type=int, default=0, help="Which round to start training (Suitable for resuming failed simulation)")
    parser.add_argument("--fed_dir", type=str, default=None, help="Provide path to use previous dataset split")
    args_parsed = parser.parse_args()

    BATCH_SIZE = args_parsed.batch_size
    RANDOM_INIT = args_parsed.random_init
    MODEL_NAME = args_parsed.model if args_parsed.load_model is None else args_parsed.load_model
    DATA = args_parsed.data
    ACCUM_STEPS = args_parsed.batch_accum
    num_rounds = args_parsed.n_iterations
    LEARNING_RATE = args_parsed.lr
    LANG_MIX = args_parsed.lang_mix
    pool_size = POOL_SIZE[args_parsed.data] # number of dataset partions (= number of total clients)
    C = args_parsed.critical_layer
    START_ROUND = args_parsed.start_round
    FED_DIR = args_parsed.fed_dir

    if args_parsed.centralized:
        pool_size = 1

    cache_str = str(args_parsed.lang_mix) + "_" + str(args_parsed.lr) + f"_{DATA}" + f"_{C}" if pool_size != 1 else f"centralized_{DATA}_" + str(args_parsed.lr) + f"_{C}"
    if args_parsed.random_init:
        cache_str += "_random"
    CACHE_STR = cache_str
    
    if args_parsed.n_gpus != 0.0:
        N_GPUS = args_parsed.n_gpus
        if num_rounds == 0:
            GPU_MAPPING["server"] = 0
        else:
            # if N_GPUS < 2 and num_rounds != 0:
            #     print(f"Given N_GPUs={N_GPUS}, need 2+ for client(s) and server to have separate GPUs. Use CPU instead otherwise")
            #     exit(1)
            num_iter_for_epoch = pool_size // (N_GPUS) if pool_size % (N_GPUS) == 0 else (pool_size // (N_GPUS)) + 1
            args_parsed.frac_fit = 1 / num_iter_for_epoch
            # TODO implement fractional GPU options if desired
            # num_rounds = int(num_rounds * num_iter_for_epoch)
            client_gpus = 1
            client_resources["num_gpus"] = client_gpus
            print(f"Using {client_gpus} GPU per client with {args_parsed.frac_fit} clients sampled per round out of {pool_size} clients")

            if torch.cuda.is_available():
                try:
                    gpus = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                except AttributeError:
                    gpus = [0]
            num_of_gpus_per_round = int(pool_size // num_iter_for_epoch)
            NUM_SKIP_EVAL = num_iter_for_epoch
            GPU_MAPPING["server"] = len(gpus) - 1
            for cid in range(pool_size):
                cid_gpu_idx = int(cid % num_of_gpus_per_round)
                GPU_MAPPING[cid] = int(gpus[cid_gpu_idx])
            print(f"GPU mapping is: {GPU_MAPPING}")

    if args_parsed.n_cpus != -1:
        client_resources["num_cpus"] = args_parsed.n_cpus

    tokenizer = make_tokenizer(args_parsed.model)
    file_path_data = DATA_TO_FILE_PATHS[args_parsed.data]
    GLOBAL_LANG_MAP = MAP_LANG_MAP[file_path_data]

    dataset_type = get_dataset_type(file_path_data)
    trainset = dataset_type(tokenizer, file_path_data, split="train", test_flag=args_parsed.test)
    split_name = "dev" if num_rounds != 0 else "test"
    print("Eval set is", split_name)
    testset = dataset_type(tokenizer, file_path_data, split=split_name, test_flag=args_parsed.test)
    if FED_DIR:
        fed_dir = FED_DIR
    else:
        fed_dir = do_fl_partitioning(
            file_path_data, trainset.examples, pool_size=pool_size, lang_mix=args_parsed.lang_mix, 
            cache_str=cache_str, val_ratio=0.2
        )

    fit_metrics_aggregation_fn: Optional[Callable] = aggregate_weighted_average
    evaluate_metrics_aggregation_fn: Optional[Callable] = aggregate_weighted_average

    # configure the strategy
    strategy = CustomFedAvgStrategy(
        fraction_fit=args_parsed.frac_fit,
        fraction_evaluate=args_parsed.frac_fit,
        min_fit_clients=pool_size,
        min_evaluate_clients=pool_size,
        min_available_clients= pool_size, #pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        evaluate_fn=get_eval_fn(testset, args_parsed.lang_mix if not args_parsed.centralized else "central")  # centralised testset evaluation of global model,
    )

    def client_fn(cid: str, optimizers=ALL_OPTIMIZERS):
        net = make_huggingface_model()
        # create a single client instance
        if cid not in optimizers:
            # Split weights in two groups, one with weight decay and the other not.
            no_decay = ["bias", "LayerNorm.weight"]
            weight_decay = 0
            learning_rate = LEARNING_RATE
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizers[cid] = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return RayClient(cid, fed_dir, optimizers[cid], net)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    # start simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args=ray_config,
    )
    save_history(hist, name = MODEL_NAME.split('/')[-1])
    ray.shutdown()