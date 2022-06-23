import cv2
import numpy as np
from keras.models import  load_model

cap = cv2.VideoCapture("testvideo.mp4")
# cap = cv2.VideoCapture(0)

# Dinh nghia class
class_name =  ['Kohaku','Ginrin', 'Goshiki', 'Hikarimuji', 'Hikarimoyo','Kumonryu', 'Kujaku', 'Doitsu', 'Chagoi',
           'Ochiba', 'Taisho Sanke', 'Showa ', 'Utsuri', 'Bekko', 'Asagi', 'Shusui', 'Tancho', 'Goromo']


# Load weights model da train
my_model = load_model('model_cc.h5')
my_model.load_weights("weight_cc.hdf5")

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
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])])
    print(np.max(predict[0],axis=0))
        # Show image
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    if (np.max(predict)>=0.3) and (np.argmax(predict[0])!=0):

        cv2.putText(image_org, class_name[np.argmax(predict)]+': '+ str(np.max(predict[0],axis=0)), org, font,
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



