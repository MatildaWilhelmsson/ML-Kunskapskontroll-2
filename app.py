import streamlit as st
import joblib
import numpy as np
from numpy import ravel

from skimage import transform
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import invert, img_as_ubyte
from skimage.filters import threshold_otsu

##Models
extra_trees = joblib.load('Extra_trees.pkl')
random_forest = joblib.load('Random_forest.pkl')
svc = joblib.load('SVC.pkl')

##Layout
st.title('Number recognition machine')
st.header('Yay!')


##Upload file
file = st.file_uploader("Load a picture!")

if file:
    st.image(file)
    file = imread(file)
    file = rgb2gray(file)
    st.image(file)
    file = invert(file)
    thresh = threshold_otsu(file)
    print("The optimal threshold is ={}.".format(thresh))
    file = (file > thresh)
    file = img_as_ubyte(file)
    st.image(file)
    file = transform.resize(file, (28, 28)).ravel()
    prediction_file = svc.predict(file.reshape(1, -1))
    st.write(prediction_file)


##Take a picture    
picture = st.camera_input("Take a picture!")

if picture:
    st.image(picture)
    picture = imread(picture)
    picture = rgb2gray(picture)
    st.image(picture)
    picture = invert(picture)
    thresh = threshold_otsu(picture)
    print("The optimal threshold is ={}.".format(thresh))
    picture = (picture > thresh)
    picture = img_as_ubyte(picture)
    st.image(picture)
    picture = transform.resize(picture, (28, 28)).ravel()
    prediction_picture = svc.predict(picture.reshape(1, -1)[0])
    st.write(prediction_picture)