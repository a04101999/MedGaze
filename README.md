This is the official implementation of MedGaze. The code will be available here.


# MedGaze

Predicting human gaze behavior within computer vision is integral for developing interactive systems that can anticipate user attention, address fundamental questions in cognitive science, and hold implications for fields like human-computer interaction (HCI) and augmented/virtual reality (AR/VR) systems. Despite methodologies introduced for modeling human eye gaze behavior, applying these models to medical imaging for scanpath prediction remains unexplored. Our proposed system aims to predict eye gaze sequences from radiology reports and CXR images, potentially streamlining data collection and enhancing AI systems using larger datasets. However, predicting human scanpaths on medical images presents unique challenges due to the diverse nature of abnormal regions. Our model predicts fixation coordinates and durations critical for medical scanpath prediction, outperforming existing models in the computer vision community. Utilizing a two-stage training process and large publicly available datasets, our approach generates static heatmaps and eye gaze videos aligned with radiology reports, facilitating comprehensive analysis. We validate our approach by comparing its performance with state-of-the-art methods and assessing its generalizability among different radiologists, introducing novel strategies to model radiologists' search patterns during CXR image diagnosis. Based on the radiologist evaluation, MedGaze can generate the human-like gaze sequences with high focus on relavant regions over the cxr-image. It sometimes also outperforms human in terms of redundancy and randomness in the scanpaths. We also provide the link for the source code and processed data in the supplementary file. 


## Human Evaluation Results 

Link for the Videos which we shared with the expert Radiologist during the Human evaluation as follows 

https://drive.google.com/drive/folders/1OCAfqEgilpSSLrkW93SkTg5ZqO5ZchZ9

Presentation slide for the Results corresponding to each video uploaded in the above link 

https://docs.google.com/presentation/d/17J15kJNDdGW86irYHKfh_GdRJaaIJ4G5EcJky8IrzHM/edit?usp=sharing


# MedGaze Pre-trained model 

## EGD-CXR 

## REFLACX

## EGD-CXR + REFLACX 

# Data links

## EGD-CXR

## REFLACX

## EGD-CXR+ REFLACX





