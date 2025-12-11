from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset

FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    # response_template_with_context = "\n### Response:"  # alpaca response tag
    # response_template_ids = tokenizer.encode(
        # response_template_with_context, add_special_tokens=False
    # )[2:

    #matched these to xTP-LLM

    data_collator= DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    #data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    return tokenizer, data_collator


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        #FederatedDataset takes additional HF load_dataset arguments like: data_files
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = client_trainset.rename_column("output", "response")

    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def load_data_from_file(partition_id: int, num_partitions: int, file_path: str):
    """Modified version of load data function above.
       Loads the data from a local file instead
   """
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:

        # finetune_lora.py line 354 - 375
        extension = (file_path.split(".")[-1])
        if extension == "txt":
            extension = "text"
        
        partitioner = IidPartitioner(num_partitions=num_partitions)

        FDS = FederatedDataset(
            dataset=extension, 
            data_files = file_path, 
            partitioners={"train": partitioner},
        )

    client_trainset = FDS.load_partition(partition_id, "train")
    return client_trainset

