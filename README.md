# COVID_xrays
This project is a Web tool for the screening of COVID-19 from chest x-ray images.

## Inside this repo:

### 1- covid_xrays_model

This is a python library to classify chest x-ray images using a deep neural network (CNN). It's
 used for training and prediction.
 

### 2- covid_xrays

A Flask GUI app in a docker container ready for deployment. 
It allows the user to upload an x-ray image and then uses the library `covid_xrays_model` to
 predict the probability of having COVID-19.
 
 ### 3- Notebooks
 
 Notebooks used for some exploratory analysis and experimentation. Only a playground for trying
  ideas.
 
 ### 4- Scripts
 
 Include scripts to generate the training data from the publicly available COVIDx dataset.
