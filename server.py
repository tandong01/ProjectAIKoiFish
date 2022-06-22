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
model = tf.keras.models.load_model("model_cc.h5") #model m train

### load file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])
video = st.file_uploader("Choose an video file", type=["mp4"])

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

if video is not None:
    cap = cv2.VideoCapture(video)
    while(True):
        # Capture frame-by-frame
        #
        ret, image_org = cap.read()
        if not ret:
            continue
        image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
        # Resize
        image = image_org.copy()
        image = cv2.resize(image, dsize=(224, 224))
        image = image.astype('float')*1./255
        # Convert to tensor
        image = np.expand_dims(image, axis=0)

        # Predict
        predict = model.predict(image)
        print("This picture is: ", classes[np.argmax(predict[0])])
        print(np.max(predict[0],axis=0))
            # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (0, 255, 0)
        thickness = 2
        if (np.max(predict)>=0.3) and (np.argmax(predict[0])!=0):

            cv2.putText(image_org, classes[np.argmax(predict)]+': '+ str(np.max(predict[0],axis=0)), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(image_org,'Unknow', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Picture", image_org)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
