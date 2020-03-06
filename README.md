In order to use this file:

- Need to be running Python 3.6
- Need to be in the l2m python environment with packages installed as instructed
- Need to have PyTorch installed in this environment

- To create your own task, inherit from PredictTotalRewardCT in l2arcadekit, follow example in l2adata.py
- Use filtering to extract background color
- Somehow (??) get this info returned in output spaces
- Discretize color

- Note: need to change environment variables because the TEF created a "workspace" folder in home that is too big



Notes: Things to change 3/3/20

- Train function should act on existing objects, not create them.
- Train function should not directly deal with hardware info (use sub functions if necessary)
- Just have functions that calculate various losses
- Model object and separate optimizer object(s) taking model parameters
- A "model" has no detach call
- Have a model aggregator class (I guess still an nn.Model) that combines separate models (like the adversaries) along with their optimizers (??)
- Specific train functions for each version
- Nice recursive ways to save models / checkpoints
- (Think about the orthogonality of the above)
- Try not to reuse code from train to test
- Use kwargs everywhere?

Re:dataloader
- Reconsider sampling model
- Predictors for other values like paddle size, to see if they work.
- How to deal with small features?


Notes: Things for 3/5/20
- Add ball color ability to dataloader (done)
- New optimizer(s) that allow for specific prediction of attributes, or combined prediction
- Check print stats to allow ball color
- Experiments with ball color vs. paddle length
