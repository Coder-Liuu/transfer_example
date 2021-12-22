import cv2
import os


dir_file = os.listdir("data/tinyimage")
for dir_ in dir_file:
    for file in os.listdir("data/tinyimage/" + dir_):
        print(file)
        img = cv2.imread(f"data/tinyimage/{dir_}/{file}", -1)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)  
        cv2.imwrite(f"data/tiny_image/{dir_}/{file}", img)




