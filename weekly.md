
## Monday, Nov 17
### Addressing issues: 
    - Wandb logging multiple clients in a flower FL simulation: Client's logs overwrite each other
    - flower simulation of xTP-LLM is missing global model aggregation step and/or metrics
    - Out of memory when running inference.py

### Solutions/ideas to address issues

#### Wandb loging
    - Wandb is initalized on each client individually, try to make it stateful, that is initialize wandb ONCE for each client. Not for each pair of round and client
    - Log the loss vs round for each client 
    - make sure to end the wandb run for each intialization, with `wandb.run()`
    - the `context.state` variable passed to each client function by flower only holds `RecordArray` objects. Find a work around in order to store a wandb run instance

#### Global flower model aggregation
    - Was only aggregating on `0.1` of the total clients each round with 3 clients. Going to try running it again but aggregating on all clients each round. 

#### inference.py
    - Ran inference.py on the entire H100 GPU instead of 1/2 of it. There is now 80 gb of VRAM. Should run fine

## Friday, Nov 21

### Tasks/Problems to Address 
    - Log evaluation metrics on wandb for global server and each individual client (federated xtp)
    - Access Federatedly trained global LLM model (Make note of how many clients trained the model and how the data was split)
    - Evaluated the global model (Inference.py) save error metrics for predictions
    - Inference.py keeps running into OOM error and crashing (even after allocating a lot more memory)

### Solutions
    - Inference.py OOM error: in `inference.sh` the shell script which calls inference.py and passes all relevant parameters passed the ouput directory of the training script as the directory of the new model. That directory contained all checkpoints from model fine-tuning, and so it (tried) loading all of those checkpoints into memory and hence ran out of it. Hopefully fixed the problem by specifying the specific checkpoint to use. Currently waiting for the job to execute.

## Monday Nov 24

### email
- [X] compute canada (OOM inference, run multiple clients on separate nodes) 
- [X] xTP-LLM Authors
- [X] Seerat
- [ ] ~flower.ai (contributer to flowertune-llm, CUDA out of memory)~

## Tuesday Nov 25
- [X] Meet with Seerat
- [X] Fix `inference.py` OOM error
- [X] Fix Flower out of VRAM error
- [X] Fix flower 'no default evaluator for client' error flower
