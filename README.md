# **_Diabetic Retinopathy Detection_**
> Information about this disease is at the bottom

#### The below video link will redirect to a detailed explanation of this project:>
>**https://1drv.ms/v/s!Avx3FKuC_L2VsklTF9VyDYxqUDRG**

# ------------------------------------------------------

#### Project Description:

- This project involves classification of a fundus image into DR class using CNN. 
- Total 4 CNN models are trained, considering this problem as **multi-label** classification and **ensembled**.
- Each model is fine-tuned on **EfficientNet**.
- The data is collected from kaggle competition: link :>
>https://www.kaggle.com/c/aptos2019-blindness-detection  
- Each model has _private score_ of kaggle of above **90%**.
- The _validation accuracy_ after **ensembling** is **93%**.
- The web-application is made using **flask**.
- Technology used: **Keras with Tensorflow backend**
>Tensorflow version 2 is used

# ------------------------------------------------------

## How to run this porject:
> **The final code directory of this project is : Diabetic_Retinopathy_Detection/src/webapp/**

> **The webapp is the interface to the ensembled model.**

- ### Note : Python 3.7.6 is used for this project

- #### Clone this repository

- #### Then run the command :
> **_pip install -r requirements.txt_**

- #### Now download the models:
> **Follow:- Diabetic_Retinopathy_Detection/src/webapp/models/README.md**
(Please upvote the models dataset if you like it)

- #### Now run the webapplication:
> **Follow:- Diabetic_Retinopathy_Detection/src/webapp/README.md**

# ------------------------------------------------------

## **About Diabetic Retinopathy**

#### Introduction

Diabetic retinopathy is the leading cause of blindness among working aged adults. Millions of people suffer from this decease. People with diabetes can have an eye disease called diabetic retinopathy. This is when high blood sugar levels cause damage to blood vessels in the retina. These blood vessels can swell and leak. Or they can close, stopping blood from passing through. Sometimes abnormal new blood vessels grow on the retina. All of these changes can lead to blindness.

#### Stages of Diabetic Eye Disease

NPDR (non-proliferative diabetic retinopathy): With NPDR, tiny blood vessels leak, making the retina swell. When the macula swells, it is called macular edema. This is the most common reason why people with diabetes lose their vision. Also with NPDR, blood vessels in the retina can close off. This is called macular ischemia. When that happens, blood cannot reach the macula. Sometimes tiny particles called exudates can form in the retina. These can affect vision too.

PDR (proliferative diabetic retinopathy): PDR is the more advanced stage of diabetic eye disease. It happens when the retina starts growing new blood vessels. This is called neovascularization. These fragile new vessels often bleed into the vitreous. If they only bleed a little, you might see a few dark floaters. If they bleed a lot, it might block all vision. These new blood vessels can form scar tissue. Scar tissue can cause problems with the macula or lead to a detached retina. PDR is very serious, and can steal both your central and peripheral (side) vision.

Source : https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy

# ------------------------------------------------------


