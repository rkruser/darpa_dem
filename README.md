In order to use this file:

- Need to be running Python 3.6
- Need to be in the l2m python environment with packages installed as instructed
- Need to have PyTorch installed in this environment

- To create your own task, inherit from PredictTotalRewardCT in l2arcadekit, follow example in l2adata.py
- Use filtering to extract background color
- Somehow (??) get this info returned in output spaces
- Discretize color

- Note: need to change environment variables because the TEF created a "workspace" folder in home that is too big
