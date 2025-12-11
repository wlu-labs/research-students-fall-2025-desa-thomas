"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer, set_seed
from trl import SFTTrainer

from flwr_xtp.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data_from_file,
    replace_keys,
)
from flwr_xtp.models import cosine_annealing, get_model

import os
import wandb

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()

wandb_runs = {}

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Parse config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_rounds = context.run_config["num-server-rounds"]
    cur_round = msg.content["config"]["server-round"]


    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    set_seed(cfg.train.seed)

    #----- Setup wandb for client -----

    if partition_id not in wandb_runs.keys():

        run = wandb.init(project = "flwr-xtp", name = f'Client:{partition_id}', config=vars(training_arguments))

        wandb_runs[partition_id] = run

    #----- end wandb setup -----


    # Let's get the client partition
    trainset = load_data_from_file(partition_id, num_partitions, cfg.dataset.name)
    
    (
    tokenizer,
    data_collator
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    # Set learning rate for current round
    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]
    
    def tokenize_fn(batch):
        output = tokenizer(batch["text"], padding=False, truncation=True)
        output['labels'] = output['input_ids'].copy()
        return output

    tokenized_trainset = trainset.map(tokenize_fn, batched=True)

    # Construct trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        # max_seq_length=cfg.train.seq_length,
        train_dataset=tokenized_trainset,
        # formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    # Do local training
    results = trainer.train()

    # Construct and return reply Message
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    # -- wandb logging --
    wandb_runs[partition_id].log({"train_loss": results.training_loss})

    if cur_round == num_rounds: 
        wandb_runs[partition_id].finish()

    return Message(content=content, reply_to=msg)



