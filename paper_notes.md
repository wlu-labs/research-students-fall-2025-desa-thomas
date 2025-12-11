# Components
1. Abstract
    - Problem statement, Objectives we aim to solve, methodology, results
    - Introduce TFP
    - Objectives
    - methodology
    - results
    
2. Introduction
    - Introduce TFP and why it is important
    - Current issues in TFP (e.g. accuracy)
    - Existing state of art traffic flow prediction
    - Why we need federated learning for traffic flow prediction(e.g., city level): generalized model that can assist us for transfer learning. If we have a model that learns from one city 
    and can be applied to a different one
    - Why is it important to apply fed to TFP 
    - This manuscript answers the following question: RQ1, RQ2, RQ3
    - The rest of the paper is organized as follows: 
    
3. Literature Review
    - History of deeplearning models for TFP (6 papers)
    - History of LLMs for traffic flow prediction
    - History of Federated Learning for TFP
    - History of FedLM for TRP
    - last 2 sentences: Gap in existing literature

4. Methodology (Finish first)
    - Data prep (Comes from xTP-LLM, then how I split that data between clients)
    - Model Architecture (overview of the high level math concepts i.e., federated average, LLM fine tuning that hugging face library uses)

    - Problem formulation (TFP) and any mathematics behind it
    - Federated learning environment creating clients
    - XTP-LLM Architecture
    - equations

5. Results and discussion (Finsh second)
    - Present your results along side what machines you have used. 
    - Creating clients and split the data
    - Results I am getting 
    - Comparison with any SOTA (state of the art model) @seerat results
    - Performance of trained model, comparison to other traffic predicting models

6. Ablation Analysis 
    - Whether added complexity is necessary/beneficial
    - Weaknesses in the study (potentially how the data was split)

9. Conclusion + Future Scope
    - Rewritten abstract (slightly more detail)
    - How you will extend this work in the future


NOTE: inference results on my trained baseline xTP are on .../xTP-LLM/eval_ouputs 
