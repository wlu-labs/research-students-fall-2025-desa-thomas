# FL-xtp

Used [flowertune-llm](https://github.com/adap/flower/tree/main/examples/flowertune-llm) boiler plate large language model fine tuning script with [xTP-LLM](https://github.com/Guoxs/xTP-LLM)framework.

## Setup
1. Clone this project using `git clone`
2. Change into `FL-xtp` directory
3. Install dependencies using `pip install -e .`
4. Make sure your dataset is installed into `./datasets` directory. We used a copy of [xTP-LLMs](https://github.com/Guoxs/xTP-LLM) mini CATraffic dataset.
5. Run the project with `flwr run ./` 
