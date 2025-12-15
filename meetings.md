# Meeting Minutes
## Thurs, Sep 11
- [x] Look into Compute Canada, and how I can access these resources
- [x] Set up FL development environment on my machine (Flower AI/ Flare)
- [x] Clone `xTP-LLM` repository

### Other Notes & Information
- Dr. Sehra has a fine tuned `xTP-LLM` model.
- Main problem with `xTP-LLM` is that it is very computationally expensive (around 25 GBs of GPU memory) 
 - Try to implement 1 or 2 papers (clone repo, run model locally). Find gaps, and bring novelty. 

## Friday, Sep 19
- ~[ ] Clone Flower ai LLM example from github~ 
- ~[ ] Run FL LLM simulation on flower~
    -Need GPU to run flowertune-llm example
### Other Notes and Information:
- Google summer of code

## Tuesday, Sep 23 
- [x] Register for compute canadas resources 

## Friday Sep 26
- [x] Access compute canadas resources from my machine (setup ssh key and request access)
- [X] Run flowertune-llm example and simulate FL setting (3 clients) on HPC resources

## Friday Oct 3
- [X] Clone and run xTP-LLM on HPC server
    - 
- [ ] Connect xTP-LLM to flower (run with 3 clients)
- [X] Connect Wand B account to xTP-LLM instance

### notes
- setup wandb


## Friday Oct 31
- [X] Initalize wandb on flower client method ()
- [X] Run xTP-LLM on Flower (don't split dataset) on 3 clients
- [ ] Run xTP-LLM on Flower with splitting dataset
- [ ] Store training metrics 

## Tuesday November 4
- [ ] Run inference_chat.py (xTP validation script) on finetuned xtp-llm (outputs/mini_datasets/openlm-research/$modelname/checkpoint...)
- [X] Updated weekly reports up to november 4th

## Tuesday November 11
- [ ] Run inference_chat.py on trained xtp-LLM 
- [ ] Review seerat's flower code and fix aggregated model metrics
- [ ] Look into fixing wandb logging for federated clients

## Friday November 14 
- [ ] Fix problem with inference_chat memory usage

## Friday Nov 21
- [ ] Get evaluation metrics for federated LLM 
