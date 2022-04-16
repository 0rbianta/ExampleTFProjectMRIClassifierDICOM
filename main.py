from random import choice as randchoice
from time import sleep, time
from PIL import ImageDraw, ImageFont
from neural import neuralRunner

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
                2) Load an DICOM Folder
                3) About
                4) Exit
                Copyright © 0rbianta / Cem Pişkinpaşa / OrxAI''')

        usr = input("prompt > ")
        if usr.isnumeric():
            usr = int(usr)
        if usr == 1:
            print("You choose the option for loading an brain MRI image. Please enter the path of an image that brains top profile easly visible.")
            usr = input("prompt > option 1 > ")
            # Run with an test image for debugging purposes
            usr = "./dataset/yes/Y4.jpg"
            print("Executing for image", usr)
            startMS = getMS()
            img = neuralRunner.loadImage(usr)
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
                    print(["    Input shape and networks shape different.",
                    "   An unknown error happened while forwarding image through the network."][int(classID)])
                    print("Please contact to the developer for help. His mail is available on the about section.")
            else:
                print(f'Failed to load image due to {["file not found", "an unknown error"][int(img)]}.')


        elif usr == 2:
            print("Not yet implemented.")
        elif usr == 3:
            print('''This will be the content of about section soon. Please wait for the development process to complete. Also there is my mail, orbianta@protonmail.com''')
        elif usr == 4 or usr == "exit" or usr == "exit()" or usr == "q":
            print(randchoice(["👋 Good Bye!", "👋 See you!", "👋 We wish you healthy days!"]))
            exit(0)
        else:
            print("🤔 Invalid command entered. Please enter a valid number listed in the list.")

        sleep(1)