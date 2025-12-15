# FL-xtp

Used [flowertune-llm](https://github.com/adap/flower/tree/main/examples/flowertune-llm) boiler plate with [xTP-LLM](https://github.com/Guoxs/xTP-LLM) framework.

## Setup
1. Clone this project using `git clone`
2. `cd` into the `FL-xtp` directory
3. Install dependencies using `pip install -e .`
4. Make sure your dataset is installed into `./datasets` directory. We used a copy of [xTP-LLMs](https://github.com/Guoxs/xTP-LLM) mini CATraffic dataset.
5. Create a shell variable `WANDB_API_KEY` with your wandb key: `export WANDB_API_KEY="your_key"`
6. Run the project with `flwr run ./` 
