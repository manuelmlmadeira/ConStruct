# Generative Modelling of Structurally Constrained Graphs




## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n construct rdkit=2023.03.2 python=3.9```
  - `conda activate construct`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```

  - Navigate to the ./ConStruct/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```


## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first before launching full experiments.
  - To run the diffusion model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=tree`. Look at `configs/dataset` for the list of datasets that are currently available
  - To reproduce the experiments in the paper, please add the flag `+experiment` to  get the correct configuration: `python3 main.py +experiment=<dataset_name>`
  - To test the obtained models, specify the path to a model with the flag `general.test_only`, it will load the model and test it, e.g., `python3 main.py +experiment=tree general.test_only=<path>`
  - The projector is controlled by the flag `model.rev_proj` (options for now: `planar`, `tree`, or `lobster`)
  - The edge-absorbing edge noise model is set through `model.transiton=absorbing_edges`.
