
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tf.keras.models import model_from_json
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# Traffic Sign Image Detection App

This app detects and classifies German Traffic Signs!

Having built a deep learning model with a >95% accuracy, this app can accurately detect over 40 different traffic signs.

""")
         
st.sidebar.header("User Input Traffic Sign Image")
         
st.sidebar.markdown("""
[Example .JPG or .PNG input file](http://bicyclegermany.com/Images/Laws/100_1607.jpg)
         """)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


@st.cache(allow_output_mutation=True)
def load_model():
    json_file = open('/Users/anasafzal/Desktop/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('/Users/anasafzal/Desktop/model.h5')     
    model = loaded_model
    return model

with st.spinner('Loading Model Into Memory....'):\
    model = load_model()

classes_dict = { 0:'Maximum Speed: 20kmph',
            1:'Maximum Speed: 30kmph', 
            2:'Maximum Speed: 50kmph', 
            3:'Maximum Speed: 60kmph', 
            4:'Maximum Speed: 70kmph', 
            5:'Maximum Speed: 80kmph', 
            6:'End of 80kmph zone', 
            7:'Maximum Speed: 100kmph', 
            8:'Maximum Speed: 120kmph', 
            9:'No Overtaking', 
            10:'No overtaking for vehicles over 3.5t', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Vehicles over 3.5 tonnes prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End of all restrictions', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'No left turn', 
            37:'No right turn', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no overtaking', 
            42:'End no overtaking for vehicles > 3.5 tonnes' }
classes = list(classes_dict.values())


def import_and_predict(image_data,model):
    size = (150,150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(30,30), interpolation=cv2.INTER_CUBIC))/255.0
    img_reshape = img_resize[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    prediction = import_and_predict(image,model)
    if 0 <= np.argmax(prediction) <= 43:
        st.write("Image detected as: "+classes_dict.get(np.argmax(prediction)))
