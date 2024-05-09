This is the official implementation of MedGaze. The code will be available here.   [under developement]


# MedGaze

Predicting human gaze behavior within computer vision is integral for developing interactive systems that can anticipate user attention, address fundamental questions in cognitive science, and hold implications for fields like human-computer interaction (HCI) and augmented/virtual reality (AR/VR) systems. Despite methodologies introduced for modeling human eye gaze behavior, applying these models to medical imaging for scanpath prediction remains unexplored. Our proposed system aims to predict eye gaze sequences from radiology reports and CXR images, potentially streamlining data collection and enhancing AI systems using larger datasets. However, predicting human scanpaths on medical images presents unique challenges due to the diverse nature of abnormal regions. Our model predicts fixation coordinates and durations critical for medical scanpath prediction, outperforming existing models in the computer vision community. Utilizing a two-stage training process and large publicly available datasets, our approach generates static heatmaps and eye gaze videos aligned with radiology reports, facilitating comprehensive analysis. We validate our approach by comparing its performance with state-of-the-art methods and assessing its generalizability among different radiologists, introducing novel strategies to model radiologists' search patterns during CXR image diagnosis. Based on the radiologist evaluation, MedGaze can generate the human-like gaze sequences with high focus on relavant regions over the cxr-image. It sometimes also outperforms human in terms of redundancy and randomness in the scanpaths. We also provide the link for the source code and processed data in the supplementary file. 


## Human Evaluation Results 

Link for the Videos which we shared with the expert Radiologist during the Human evaluation as follows 

https://drive.google.com/drive/folders/1OCAfqEgilpSSLrkW93SkTg5ZqO5ZchZ9

Presentation slide for the Results corresponding to each video uploaded in the above link 

https://docs.google.com/presentation/d/17J15kJNDdGW86irYHKfh_GdRJaaIJ4G5EcJky8IrzHM/edit?usp=sharing


# Data links for Training and Testing MedGaze


## ResNet50 CXR Image features link ( EGD-CXR + REFLACX )

https://drive.google.com/drive/folders/1YZCgSziKLH-4-as9wIZExzDLjDJ6My8D?usp=sharing



# Radiology Reports and processed Eye Gaze Data ( Resize accordingly based on image size- This is a raw data )

## EGD-CXR

Full dataset [train +test]

https://drive.google.com/file/d/1hdiBszZHc04HazbuNZlh90DyrsvQO9of/view?usp=sharing

Train file used in paper 

https://drive.google.com/file/d/1VXFEPXKAEw-eFdeXumeIwl_xcWOv0Xi9/view?usp=sharing

Test file used in paper 

https://drive.google.com/file/d/10ah8HmFi2XSr0RRxTjXeKo5Bb7hmA78r/view?usp=sharing

## REFLACX

Full dataset [train +test]

https://drive.google.com/file/d/1CBQ3YZ7tm3yKYuMXEIStdaL-m-qsiNa8/view?usp=sharing

Train file used in paper 

https://drive.google.com/file/d/10M_6P9VUK8lC727vBctBJqQgriroWvpl/view?usp=sharing

Test file used in paper

https://drive.google.com/file/d/1CvM0ME6gEy6APP20YEGvtTsxgViL5Khw/view?usp=sharing

## EGD-CXR+ REFLACX


Full dataset [train +test]

https://drive.google.com/file/d/1Ls5zzLCRjDBJJP4JSoxtElHoy7bDQZVA/view?usp=sharing


Train file used in paper 

https://drive.google.com/file/d/1NcCoYtKYfIN8Re4Ez5396AfTRB7nB3tB/view?usp=sharing



Test file used in paper

https://drive.google.com/file/d/1Hnf_671_suI6_fL1dOCS7A5mlXBm3Coe/view?usp=sharing

# MedGaze Pre-trained model 

## EGD-CXR 

https://drive.google.com/file/d/1cXa-LF97RWGHzB2yBPq8-W_m3FTQEkZy/view?usp=sharing

## REFLACX

## EGD-CXR + REFLACX 

# Training MedGaze 

Please update the medgaze_training.py file for the desired arguments as below:

training file  path

testing  file path

image features [Resnet50] path

Model directory to save results 

To start the training of medgaze run 

```

python Medgaze_training.py

```



# Testing MedGaze on the dataset 


All the results corresponding to each dataset can be reproduced using the notebook provided 

Notebook_miccai.ipynb



