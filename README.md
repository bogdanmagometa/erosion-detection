# Soil erosion detection

## Prerequisites
- Nvidia GPU
- conda

## Project structure
- analysis.ipynb - Jupyter notebook with analysis
- train.py - Python module that trains the model (currently fitted weights are not saved)
- README.md
- requirements.txt - Python libraries for installing with pip
- SolutionReport.pdf - solution report with suggestions
- model.png - file with model architecture

## Environment setup
Run the following in the project root directory to set up conda environment:
```bash
$ conda create -n erosion_detection python=3.9  # create new virtual env
$ conda activate erosion_detection              # activate environment in terminal
$ conda install jupyter                         # install jupyter + notebook
$ conda install tensorflow-gpu                  # install tensorflow with GPU support
$ pip install -r requirements.txt               # install python libraries used in analysis.ipynb and train.py
```

