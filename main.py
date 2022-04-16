from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

net = load_model("./keras_model.h5")


def main():
    path = input("IMG PATH>> ")
    path = "./archive/yes/Y4.jpg"
    img = load_img(
        path, target_size=(224, 224, 3))
    np.array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    network_output = net.predict(img, batch_size=1)
    print(np.argmax(network_output))


if __name__ == "__main__":
    print('''Welcome to AI Brain MRI Tumor Detector App
            1) Load an Brain MRI Image
            2) Load an DICOM Folder
            3) About
            4) Exit
            Copyright © 0rbianta / Cem Pişkinpaşa / OrxAI''')
    
    # App main loop
    while True:
        usr = input("prompt > ")
        if usr == 1:
            print("You choose the option for loading an brain MRI data image. Please enter the path of the image.")
            usr = input("prompt > ")


        elif usr == 2:
            pass
        elif usr == 3:
            pass
        elif usr == 4:
            pass
        else:
            print("Invalid command entered. Please enter a valid number listed in the list")