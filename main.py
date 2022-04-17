import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from colorama import Fore
from random import choice as randchoice
from time import sleep, time
from PIL import ImageDraw, ImageFont
import PIL
import numpy as np
from neural import neuralRunner
from pydicom import dcmread


neuralRunner = neuralRunner("./keras_model.h5", "labelmap.txt")
imgDrawFont = "./Anonymous_Pro/AnonymousPro-Regular.ttf"
getMS = lambda : round(time() * 1000)


def showImageOutput(img, label: str):
    draw = ImageDraw.Draw(img)
    draw.text((3, 3), "Output: " + label, (255, 0, 0), 
        font=ImageFont.truetype(imgDrawFont, 17))
    draw.text((3, 20), "Copyright © 0rbianta", (255, 0, 0), 
        font=ImageFont.truetype(imgDrawFont, 13))
    img.show()


if __name__ == "__main__":
    # App main loop
    while True:
        print('''Welcome to AI Brain MRI Tumor Detector App
                1) Load an Brain MRI Image
                2) Load an DICOM File
                3) Dump DICOM Data
                4) About
                5) Exit
                Copyright © 0rbianta / Cem Pişkinpaşa / OrxAI''')

        usr = input("prompt > ")
        if usr.isnumeric():
            usr = int(usr)
        if usr == 1:
            print("You choose the option for loading an brain MRI image. Please enter the path of an image that brains top profile easly visible.")
            usr = input("prompt > option 1 > ")
            print("Executing for image", usr)
            startMS = getMS()
            img = neuralRunner.loadImageFromPath(usr)
            if img not in ("0", "1"):
                classID = neuralRunner.predictWithImage(img)
                if classID not in ("0", "1"):
                    print(f"Process complete in {getMS() - startMS} milliseconds.")
                    outLabel = neuralRunner.labels[classID]
                    print(f"ClassID: {classID}, Label: {outLabel}")
                    showImageOutput(img, outLabel)
                else:
                    print("Failed to forward image through the network.")
                    print("Problem:")
                    print(["    Input shape and networks shape are different.",
                    "   An unknown error happened while forwarding image through the network."][int(classID)])
                    print("Please contact to the developer for help. His mail is available on the about section.")
            else:
                print(f'Failed to process image due to {["file not found", "an unknown error"][int(img)]}.')


        elif usr == 2:
            print("You choose the option for loading an brain MRI DICOM file. Please enter the path of an DICOM that brains top profile easly visible.")
            usr = input("prompt > option 2 > ")
            # Run with an test dicom for debugging purposes
            # usr = "./MRIDATA/DICOM/ST000000/SE000000/MR000003"
            print("Executing for DICOM", usr)
            startMS = getMS()
            try:
                dicom = dcmread(usr)
                print("Patient Age:", dicom.PatientAge)
                print("Patient Name:", dicom.PatientName)
                print("Patient Sex:", dicom.PatientSex)
                print("Study Date:", dicom.StudyDate)
                print("Study ID:", dicom.StudyID)
                
                img = PIL.Image.fromarray(dicom.pixel_array.astype(np.uint8))
                img = img.convert("RGB")
                img.thumbnail((224, 244), PIL.Image.Resampling.LANCZOS)
                
                classID = neuralRunner.predictWithImage(img)
                if classID not in ("0", "1"):
                    print(f"Process complete in {getMS() - startMS} milliseconds.")
                    outLabel = neuralRunner.labels[classID]
                    print(f"ClassID: {classID}, Label: {outLabel}")
                    showImageOutput(img, outLabel)
                else:
                    print("Failed to forward image through the network.")
                    print("Problem:")
                    print(["    Input shape and networks shape are different.",
                    "   An unknown error happened while forwarding image through the network."][int(classID)])
                    print("Please contact to the developer for help. His mail is available on the about section.")
            except FileNotFoundError:
                print("Failed to load DICOM due to file being unable to found at the given path.")
            except:
                print("Failed to continue processing DICOM due to an unknown error.")
        elif usr == 3:
            print("Enter the path of and DICOM file.")
            usr = input("prompt > option 3 > ")
            try:
                dicom = dcmread(usr)
                print("Patient Age:", dicom.PatientAge)
                print("Patient Name:", dicom.PatientName)
                print("Patient Sex:", dicom.PatientSex)
                print("Patient Size", dicom.PatientSize)
                print("Patient Weight", dicom.PatientWeight)
                print("Series UID", dicom.SeriesInstanceUID)

                print("Study Date:", dicom.StudyDate)
                print("Study ID:", dicom.StudyID)
                
                img = PIL.Image.fromarray(dicom.pixel_array.astype(np.uint8))
                img = img.convert("RGB")
                img.thumbnail((224, 244), PIL.Image.Resampling.LANCZOS)
                img.show()

            except FileNotFoundError:
                print("Failed to load DICOM due to file being unable to found at the given path.")
            except:
                print("Failed to continue processing DICOM due to an unknown error.")


        elif usr == 4:
            print('🥳 Hello there!\nI\'m Cem Pişkinpaşa, also known as 0rbianta and /dev/null on internet. ' \
            'Lastly I were having some possible neurological problems so I spent some time on the hospital. ' \
            'My neurological peak went nice. I glad that my brain and body were working together very well and responsing to all the tests doctor applied. ' \
            'After passing the tests successfully an MRI image also required so I spent some time on closed MRI too. ' \
            'I tought that all of this time and effort I spent on the hospital went to nothing because my body were doing so well so after getting the MRI images ' \
            'I went back to home and decided to create an AI that be able to decide am I healthy or not by my brain MRI images. The hopitals were give me the images on ' \
            'a DICOM file format and a software be able to read them on a DVD. I pass that files to my workspace and started to programming. Found an MRI dataset over the internet ' \
            'and trained the Google\'s most light neural network model, MobileNetv2. Fastly made a DICOM reader too and guess what. My AI were working faster then the whole hospital and I ' \
            'were already got my results before the doctor tells me. Welcome to the future!')
            print(Fore.RED + 'Third-party contents')
            print(Fore.BLACK + 'Anonymous_Pro -> https://fonts.google.com/specimen/Anonymous+Pro')
            print('Brain MRI Dataset -> https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection')
            print(Fore.BLUE + 'Contact me')
            print(Fore.BLACK + 'Cem Pişkinpaşa(He/him)')
            print('Mail: orbianta@protonmail.com')
            print('GitHub: https://github.com/0rbianta')
        elif usr == 5 or usr == "exit" or usr == "exit()" or usr == "q":
            print(randchoice(["👋 Good Bye!", "👋 See you!", "👋 We wish you healthy days!"]))
            exit(0)
        else:
            print("🤔 Invalid command entered. Please enter a valid number listed in the list.")

        sleep(1)