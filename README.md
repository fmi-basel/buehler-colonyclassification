# buehler-colonyclassification
Pipeline for automated yeast colony classification using neural networks.

## Installation

### Prerequisites:

You must have python v. 3.6 or above and the conda package manager installed. The recommended way to get this is through the Anaconda distribution - instructions are available here: https://conda.io/docs/user-guide/install/download.html

The neural networks used in the pipeline are built using the `fastai` library. You can follow the steps here to install it using conda and create a conda environment: https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652

**Important!** The colony classification pipeline runs with `fastai` **version 0.7**, which is not the latest version. Please make sure you download v0.7 and not v.1.x. This code has **not** been tested with later versions!

### Downloading and setting up the pipeline:

Download the pipeline by cloning the repository:

```git clone https://github.com/fmi-basel/buehler-colonyclassification.git
```

The script needs to be able to find the `fastai` modules that you have installed. The simplest way to ensure this is to put a symbolic link to the `fastai` directory in the top level of the `buehler-colonyclassification` directory:

```cd buehler-colonyclassification
   ln -s /path/to/fastai ./ 
```

Test that everything is working by calling up the help menu:

```python integrated_segmentation_classification.py --help
```


## Running the pipeline


