# buehler-colonyclassification
Pipeline for automated yeast colony classification using neural networks.

## Installation

### Prerequisites:

You must have python v. 3.6 or above and the conda package manager installed. The recommended way to get this is through the Anaconda distribution - instructions are available here: https://conda.io/docs/user-guide/install/download.html

The neural networks used in the pipeline are built using the `fastai` library. You can follow the steps here to install it using conda and create a conda environment: https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652

**Important!** The colony classification pipeline runs with `fastai` **version 0.7**, which is not the latest version. Please make sure you download v0.7 and not v.1.x. This code has **not** been tested with later versions!

### Downloading and setting up the pipeline:

Download the pipeline by cloning the repository:

```
git clone https://github.com/fmi-basel/buehler-colonyclassification.git
```

The script needs to be able to find the `fastai` modules that you have installed. The simplest way to ensure this is to put a symbolic link to the `fastai` directory in the top level of the `buehler-colonyclassification` directory:

```
cd buehler-colonyclassification
ln -s /path/to/fastai ./ 
```

Test that everything is working by calling up the help menu:

```
conda activate fastai
python integrated_segmentation_classification.py --help
```


## Running the pipeline

First, make sure that the images you want to process are in a readable folder. Images should be in the .jpg format. The path to this directory is the only required argument. 

Basic usage:

```
python integrated_segmentation_classification.py --destdir "/path/to/my/images"
```

You can either run the whole pipeline (colony segmentation plus classification), or run each step separately (e.g. in case you have pre-segmented images). The default is to run the full pipeline. To run only one step, add the `--predict` or `-p` flag. Allowed values are "full" for the full pipeline, "seg" for segmentation only, or "freq" for classification frequencies only.

```
python integrated_segmentation_classification.py --destdir "/path/to/my/images" --predict seg 
python integrated_segmentation_classification.py --destdir "/path/to/my/images" --predict freq
```

## Output

A successful run of the full pipeline will create two main outputs: a folder called "cropped" within the original image folder containing cropped images of each individual colony identified by the segmentation, and a CSV file named with the date and time of completion containing a summary of the classification results. The CSV file can be opened with Excel, and it contains one row per original plate image. The columns are as follows:

* bad_seg: Count of the number of colonies that were badly segmented (e.g. too small or multiple colonies per cropped image); these are excluded from further analysis.
* pink: Count of the number of predicted pink colonies
* red: Count of the number of predicted red colonies
* var: Count of the number of predicted variegating colonies
* white: Count of the number of predicted white colonies
* Perc_white: Percentage of white colonies on the plate. Defined as (white) / (white + pink + red + var).
* Perc_non_white: Percentage of non-white colonies on the plate. Defined as (pink + red + var) / (white + pink + red + var).

In general the percentage of non-white colonies is more accurate than individual predictions of red, pink, or variegating colonies. 

Please note that exact numbers may vary if the pipeline is run multiple times on the same data, as the neural network predictions are not deterministic.



