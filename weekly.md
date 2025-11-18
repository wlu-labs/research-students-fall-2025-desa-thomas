
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
