from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

class neuralRunner:
    def __init__(self, modelPath, labelPath):
        self.net = load_model(modelPath)
        self.__load_labels(labelPath)
    
    def __load_labels(self, lpath):
        file = open(lpath, "r")
        self.labels = file.read().split("\n")
        file.close()

    def loadImageFromPath(self, imgPath):
        try:
            return load_img(imgPath, target_size=(224, 224, 3))
        except FileNotFoundError:
            return "0"
        except:
            return "1"


    def predictWithImage(self, img):
        np.array(img)
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])
        try:
            net_output = self.net.predict(img, batch_size=1)
            return np.argmax(net_output)
        except ValueError:
            return "0"
        except:
            return "1"


