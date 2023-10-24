# Vision Transformer-Based Multi-Label Survival Prediction for Oropharynx Cancer Radiotherapy Using Planning CT

In this project we developed an transformer based multi-label survival prediction modeling used a publicly available dataset **OPC Radiaomics**. The dataset can be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=33948764 which contains both clincal and image data of 600+ OPC cancer patients treated in Canada. More details about the dataset can be found in:
- *Kwan JYY, Su J, Huang SH, Ghoraie LS, Xu W, Chan B, Yip KW, Giuliani M, Bayley A, Kim J, Hope AJ, Ringash J, Cho J, McNiven A, Hansen A, Goldstein D, de Almeida JR, Aerts HJ, Waldron JN, Haibe-Kains B, O'Sullivan B, Bratman SV, Liu FF.  (2018)  Radiomic Biomarkers to Refine Risk Models for Distant Metastasis in HPV-related Oropharyngeal Carcinoma . International Journal of Radiation Oncology*Biology*Physics, 1-10. DOI: 10.1016/j.ijrobp.2018.01.057* 


In the shared folder, we provided all our codes for this project, from data preprocessing to data analysis. We will go through all these sections in the following.

## Preprocessing
### Image preprocessing
We did the image preprocessing with JupyterNotebook, the file is saved at *T:\Physics_Research\LABS\LAB_Wang_Zhang\MeixuChen\OPC_MultiLabel_OutcomePrediction\data preprocessing*. It contains:
1. **Data format transformation**: we used *dcmrtstruct2nii* package to convert dicom image and RTstructure files to nii format, which is widely used in python based medical image processing open source projects. Data will be loaded from *oriPath* and saved as nii data in *savePath*. Those who don't have either CT or RTs file will be reported, and the corresponding data won't be used in the following steps. Please note, for some of the patients, their GTVp mask might not be named as *mask_GTV.nii.gz*, I modified them manually.
2. **Data availablity check**: we checked if gtvp mask is provided, if not, we won't use the corresponding patient data in the following steps.
3. **Resampling**: we resample the data saved in nii files to voxel size of 2,2,2mm. The data will be saved in *resample_path*. Please modify the value of *resampling* if you need other voxel size.
4. **Cropping**: we cropped the resampled images to the matrix size we need, which is [80,80,48] in our experiment. The center of the cropped region will be the center of GTVp mask, the image size can be set with *patch_size*. Generated images files will be saved at *crop_path* in nii format.

### Clinical data preprocessing
We did the clinical data preprocessing manually, for feature selection and feature coding, please refert our manuscript, which is saved at *T:\Physics_Research\LABS\LAB_Wang_Zhang\MeixuChen\OPC_MultiLabel_OutcomePrediction\manuscript*. The final clinical data sheet is saved at *T:\Physics_Research\LABS\LAB_Wang_Zhang\MeixuChen\OPC_MultiLabel_OutcomePrediction\data*.

### Model construction
We used pytorch-lightning for organizing our coding structure. More details of it can be found at the [official website](https://lightning.ai/). The key items include
1. Dataset
2. Model
3. Trainer
4. logger
5. others...

Details of model structures can be found in the manuscript.
### Data analysis
KM analysis, AUC calculation
JupterNotebook files saved at *T:\Physics_Research\LABS\LAB_Wang_Zhang\MeixuChen\OPC_MultiLabel_OutcomePrediction\data analysis*
Copy all the five folds validation results to a folder, then copy the analysis code to it, modify the datasheet name in the code, run all to generate the results.

