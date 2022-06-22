import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image
st.header("""
         # PHÂN LOẠI 18 LOẠI CÁ KOI
         """
         )
st.write("18 loại cá Koi bao gồm các loại: Kohaku, Ginrin, Goshiki', Hikarimuji, Hikarimoyo, Kumonryu, Kujaku, Doitsu, Chagoi,Ochiba, Taisho Sanke, Showa , Utsuri, Bekko, Asagi, Shusui, Tancho, Goromo ")
model = tf.keras.models.load_model("model_cc.h5")

### load file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])


classes = ['Kohaku', 'Ginrin', 'Goshiki', 'Hikarimuji', 'Hikarimoyo', 'Kumonryu', 'Kujaku', 'Doitsu', 'Chagoi',
               'Ochiba', 'Taisho Sanke', 'Showa ', 'Utsuri', 'Bekko', 'Asagi', 'Shusui', 'Tancho', 'Goromo']
 
if uploaded_file is not None:
    # Convert the file
    img = image.load_img(uploaded_file,target_size=(224,224)) 
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    img = img.astype('float32')
    img = img/255
        
    #Button: nút dự đoán sau khi up ảnh
    Genrate_pred = st.button("Dự đoán") 
    
    if Genrate_pred:
    
        prediction = model.predict(img)
        st.write("""Kết quả dự đoán của hình này là: {}""".format(classes [np.argmax(prediction[0])])) 
        st.write("Độ chính xác là: {:.2f} %".format(100*np.max(prediction[0],axis=0)))


