from fastai.conv_learner import *
from fastai.dataset import *
from fastai.imports import *
from fastai.transforms import *
from fastai.model import *
from fastai.sgdr import *
from fastai.plots import *
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import sys, argparse
import numpy as np

from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage.io import *
from skimage.morphology import *
from skimage.measure import *
from PIL import Image
from pathlib import Path

def main():

    parser = get_parser()

    MODELPATH = Path('./pretrained/')

    my_args=parser.parse_args()

    IOPATH=Path(my_args.iopath)
    predict=my_args.predict

    if predict == 'full':
        predict_seg(MODELPATH, IOPATH)
        predict_freqs(MODELPATH, IOPATH)
    elif predict == 'seg':
        predict_seg(MODELPATH, IOPATH)
    elif predict == 'freq':
        predict_freqs_only(MODELPATH, IOPATH)


def get_parser():

    parser = argparse.ArgumentParser(description='Run automated colony classification')
    parser.add_argument('-d', '--destdir', dest='iopath', metavar='/path/to/images',
                        help='Please enter the full path to the directory containing your images',
                        required=True, )
    parser.add_argument('-p', '--predict', dest='predict', default='full', metavar='full',
                        choices=['full', 'seg', 'freq'],
                        help='Choose whether to run the full pipeline or to predict only segmentation mask or only class frequencies. Allowed options are "full", "seg" or "freq".')
    return parser


def predict_seg(MODELPATH, IOPATH):

    trn_x = list(sorted((MODELPATH/'segmentation/train').iterdir()))
    trn_y = list(sorted((MODELPATH/'segmentation/train_mask').iterdir()))
    val_x = list(sorted((MODELPATH/'segmentation/val').iterdir()))
    val_y = list(sorted((MODELPATH/'segmentation/val_mask').iterdir()))
    test_x = list(IOPATH.iterdir())


    class MatchedFilesDataset(FilesDataset):
        def __init__(self, fnames, y, transform, path):
            self.y=y
            assert(len(fnames)==len(y))
            super().__init__(fnames, transform, path)
        def get_y(self, i):
            return open_image(os.path.join(self.path, self.y[i]))
        def get_c(self): return 0


    aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
                RandomFlip(tfm_y=TfmType.CLASS),
                RandomLighting(0.05, 0.05)]

    # U-net

    f = resnet34
    cut,lr_cut = model_meta[f]

    def get_base():
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers)

    def dice(pred, targs):
        pred = (pred>0).float()
        return 2. * (pred*targs).sum() / (pred+targs).sum()

    class SaveFeatures():
        features=None
        def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
        def hook_fn(self, module, input, output): self.features = output
        def remove(self): self.hook.remove()

    class UnetBlock(nn.Module):
        def __init__(self, up_in, x_in, n_out):
            super().__init__()
            up_out = x_out = n_out // 2
            self.x_conv = nn.Conv2d(x_in, x_out, 1)
            self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
            self.bn = nn.BatchNorm2d(n_out)

        def forward(self, up_p, x_p):
            up_p = self.tr_conv(up_p)
            x_p = self.x_conv(x_p)
            cat_p = torch.cat([up_p, x_p], dim=1)
            return self.bn(F.relu(cat_p))


    class Unet34(nn.Module):
        def __init__(self, rn):
            super().__init__()
            self.rn = rn
            self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
            self.up1 = UnetBlock(512, 256, 256)
            self.up2 = UnetBlock(256, 128, 256)
            self.up3 = UnetBlock(256, 64, 256)
            self.up4 = UnetBlock(256, 64, 256)
            self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

        def forward(self, x):
            x = F.relu(self.rn(x))
            x = self.up1(x, self.sfs[3].features)
            x = self.up2(x, self.sfs[2].features)
            x = self.up3(x, self.sfs[1].features)
            x = self.up4(x, self.sfs[0].features)
            x = self.up5(x)
            return x[:, 0]

        def close(self):
            for sf in self.sfs: sf.remove()

    class UnetModel():
        def __init__(self,model,name='unet'):
            self.model,self.name = model,name

        def get_layer_groups(self, precompute):
            lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
            return lgs + [children(self.model)[1:]]


    # 1024x1024 images

    sz=1024
    bs=2
    lr=4e-2
    wd=1e-7

    tfms = tfms_from_model(resnet34, sz, crop_type=CropType.CENTER, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
    datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), test=(test_x,test_x), tfms=tfms, path=MODELPATH/'segmentation')
    md = ImageData(MODELPATH/'segmentation', datasets, bs, num_workers=16, classes=None)
    denorm = md.trn_ds.denorm

    m_base = get_base()
    m = to_gpu(Unet34(m_base))
    models = UnetModel(m)

    learn = ConvLearner(md, models)
    learn.opt_fn=optim.Adam
    learn.crit=nn.BCEWithLogitsLoss()
    learn.metrics=[accuracy_thresh(0.5),dice]

    learn.load('1024urn_20x12')

    py, ay = learn.predict_with_targs(is_test=True)


    # Futher image processing steps

    def preprocess_image(fn, pn):
        img_pre = imread(fn)
        ysize, xsize, chan = img_pre.shape
        
        # Center crop
        if ysize == xsize:
            cropped = img_pre
            dim = ysize
        else:
            mindim = np.argmin([ysize, xsize])
            dim = min(ysize, xsize)
            if mindim == 0: 
                diff = xsize - ysize
                if diff % 2 == 0:
                    offset = diff // 2
                    cropped = img_pre[:,offset:-offset]
                else:
                    offset1 = diff // 2
                    offset2 = offset1 + 1
                    cropped = img_pre[:,offset1:-offset2]
            elif mindim == 1:
                diff = ysize - xsize
                if diff % 2 == 0:
                    offset = diff // 2
                    cropped = img_pre[offset:-offset,:]
                else:
                    offset1 = diff // 2
                    offset2 = offset1 + 1
                    cropped = img_pre[offset1:-offset2,:]        
          
         # morphological preprocessing
        mask_upscaled = resize(pn, (dim, dim), preserve_range=True)
        bool_cleared = clear_border(mask_upscaled>0)
        morphFilt = opening(bool_cleared, disk(8))
        label_reg = label(morphFilt)
        props_reg = regionprops(label_reg)
        return(cropped, dim, props_reg)


    # Set output directory name
    CROPPATH = IOPATH/'cropped'
    if not os.path.exists(CROPPATH):
        os.makedirs(CROPPATH)

    n_img = py.shape[0]

    for i in range(0,n_img):
        k = 0
        cropped, dim, props_reg = preprocess_image(test_x[i], py[i])
        my_regs = [region for region in props_reg if region.eccentricity <= 0.6 and region.area >= 400]
        for region in my_regs:
            k += 1
            minr, minc, maxr, maxc = region.bbox
            crop = cropped[minr:maxr, minc:maxc, :]
            imname = test_x[i].stem + "_" + str(k) + ".jpg"
            plt.imsave(CROPPATH/imname, crop)


## Now move onto classification with resnet34

def predict_freqs(MODELPATH, IOPATH):

    sz=50
    arch=resnet34
    tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
    test_path = IOPATH/'cropped'  # Look for the cropped folder created by the previous step

    data = ImageClassifierData.from_paths(MODELPATH/'classification', tfms=tfms, bs=128, test_name=test_path)
    learn = ConvLearner.pretrained(arch, data, precompute=False)
    learn.unfreeze()
    learn.load("multi5class_adaptive_lr") # best model so far - includes a class for multi-colony images

    # with TTA
    log_pred_test, y_test = learn.TTA(is_test=True)
    probs_test = np.mean(np.exp(log_pred_test),0)
    preds_test = np.argmax(probs_test, axis=1)

    test_filenames = data.test_ds.fnames

    filenames = [ ]
    for name in test_filenames:
        m = re.search('(.+)_(\d+).jpg$', name)
        if m:
            filenames.append(m.groups()[0])

    filenames = np.asarray(filenames, dtype=np.str)
    unique_filenames = np.unique(filenames)


    # Format output data table

    full_df = pd.DataFrame()
    for name in np.nditer(unique_filenames):
        print(name)
        ind = []
        for tname in np.nditer(filenames):
            ind.append(tname == name)
        f_preds = preds_test[ind]
        #print(f_preds)
        f_vals, f_counts = np.unique(f_preds, return_counts=True)

        f_ser = pd.Series(f_counts, index=f_vals)
        f_df = pd.DataFrame(f_ser, index=[0,1,2,3,4], columns=[str(name)])
        full_df = full_df.append(f_df.T)

    full_df.columns = ['bad_seg', 'pink', 'red', 'var', 'white']
    full_df = full_df.assign(Perc_white = full_df['white']/full_df.iloc[:,1:5].sum(axis=1), Perc_non_white = full_df.iloc[:,1:4].sum(axis=1)/full_df.iloc[:,1:5].sum(axis=1))

    import datetime
    timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    summary_filename = timestamp + '_summary.csv'

    full_df.to_csv(IOPATH/summary_filename, na_rep="0")



def predict_freqs_only(MODELPATH, IOPATH):

    sz=50
    arch=resnet34
    tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
    test_path = IOPATH

    data = ImageClassifierData.from_paths(MODELPATH/'classification', tfms=tfms, bs=128, test_name=test_path)
    learn = ConvLearner.pretrained(arch, data, precompute=False)
    learn.unfreeze()
    learn.load("multi5class_adaptive_lr") # best model so far - includes a class for multi-colony images

    # with TTA
    log_pred_test, y_test = learn.TTA(is_test=True)
    probs_test = np.mean(np.exp(log_pred_test),0)
    preds_test = np.argmax(probs_test, axis=1)

    test_filenames = data.test_ds.fnames

    filenames = [ ]
    for name in test_filenames:
        m = re.search('(.+)_(\d+).jpg$', name)
        if m:
            filenames.append(m.groups()[0])

    filenames = np.asarray(filenames, dtype=np.str)
    unique_filenames = np.unique(filenames)


    # Format output data table

    full_df = pd.DataFrame()
    for name in np.nditer(unique_filenames):
        print(name)
        ind = []
        for tname in np.nditer(filenames):
            ind.append(tname == name)
        f_preds = preds_test[ind]
        #print(f_preds)
        f_vals, f_counts = np.unique(f_preds, return_counts=True)

        f_ser = pd.Series(f_counts, index=f_vals)
        f_df = pd.DataFrame(f_ser, index=[0,1,2,3,4], columns=[str(name)])
        full_df = full_df.append(f_df.T)

    full_df.columns = ['bad_seg', 'pink', 'red', 'var', 'white']
    full_df = full_df.assign(Perc_white = full_df['white']/full_df.iloc[:,1:5].sum(axis=1), Perc_non_white = full_df.iloc[:,1:4].sum(axis=1)/full_df.iloc[:,1:5].sum(axis=1))

    import datetime
    timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    summary_filename = timestamp + '_summary.csv'

    full_df.to_csv(IOPATH/summary_filename, na_rep="0")



if __name__ == "__main__":
    main()
