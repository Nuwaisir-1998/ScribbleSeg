# ScribbleSeg
A method to segment Spatial Transcriptomics data.

# Prerequisites
Results are generated using Google Colab Standard GPU with:
```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
These lines ensure reproducible results

Python version: 3.10.6

# How to run?
At first you have to run preprocessor.py. For expert scribble scheme run the following:
```
python preprocessor.py --scheme expert
```
For mclust scribble scheme run the following:
```
python preprocessor.py --scheme mclust
```
Then, to generate the segmentations for expert scribble scheme, run:
```
python expert_scribble_pipeline.py
```
Or, to generate the segmentations for mclust scribble scheme, run:
```
python mclust_scribble_pipeline.py
```
Results will be generated in Outputs directory.

# Other Informations
The folder 'Supplementary_figures' has the figures of the supplementary information of our research paper
