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

# Input
Set the input parameters in Inputs/[file_name].json

# How to generate scribbles?
Scribbles can be generated using [Loupe browser](https://support.10xgenomics.com/single-cell-gene-expression/software/visualization/latest/what-is-loupe-cell-browser)

# How to run?
First set your environment by running:
```
conda env create -f environment.yml
```
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
python expert_scribble_pipeline.py --params ./Inputs/expert_scribble_scheme_input.json
```
Or, to generate the segmentations for mclust scribble scheme, run:
```
python expert_scribble_pipeline.py --params ./Inputs/mclust_scribble_scheme_input.json
```
Results will be generated in Outputs directory.

# Other Informations
The folder 'Supplementary_figures' has the figures of the supplementary information of our research paper
