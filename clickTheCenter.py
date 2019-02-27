from tkinter import *
from tkinter import ttk
import sys
sys.dont_write_bytecode = True
import os
import os.path
from classDefinitions import *
from PIL import Image
import numpy as np



#### Globals ###############################
selectedObjectType = classes['unspecified']
filelist = []
imageLabel = None
currentImageFilename = ''
currentImageData = None

if(len(sys.argv) > 1):
    directory=sys.argv[1]
else:
    directory=""


#### Functions ###############################

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def testImage(filename):
    global currentImageData
    if not os.path.isfile(filename):
        return False

    image = Image.open(filename)

    desiredShape = (480, 640, 3) # we only want 3 color channels, because we add the fourth
    currentImageData = np.array(image)
    return currentImageData.shape == desiredShape

def nextImage(*args):
    global directory, filelist, currentImageFilename
    print("next")
    goodImage = False
    while not goodImage:
        if len(filelist) == 0:
            print("All images in folder processed, quitting")
            quitprog()
        nextImage = filelist.pop()
        currentImageFilename = os.path.join(directory, nextImage)
        goodImage = testImage(currentImageFilename)

    padImageData()
    print("showing", currentImageFilename, " ", len(filelist), "images left")
    displayImage(currentImageFilename);

def padImageData():
    global currentImageData
    imageDims = (currentImageData.shape[0], currentImageData.shape[1], 1)
    alpha = np.full(imageDims, classes['unspecified'])
    currentImageData = np.concatenate((currentImageData, alpha), axis=2)


def undo(*args):
    global currentImageData
    alphaChannel = currentImageData[:,:,3]
    alphaChannel.fill(classes['unspecified'])

def saveAndNext(*args):
    #TODO warn if no pixel selected
    img = Image.fromarray(currentImageData.astype(np.uint8))
    img.save(currentImageFilename)
    nextImage()

def displayImage(filename):
    global imageLabel
    photo = PhotoImage(file=filename)
    imageLabel.configure(image = photo)
    imageLabel.image = photo

def quitprog(*args):
    sys.exit(0)

def onclick(event):
    global currentImageData, selectedObjectType
    print("set pixel at", event.x, event.y, "to be", selectedObjectType)
    currentImageData[event.y, event.x, 3] = selectedObjectType

def selectObjectType(objType):
    global selectedObjectType
    selectedObjectType = objType

def loadFilelist():
    global filelist, directory
    filelist = sorted(os.listdir(directory))

def deleteImage(*args):
  os.remove(currentImageFilename)
  nextImage()

def main():
    global selectedObjectType, imageLabel

    loadFilelist()

    root = Tk()
    root.attributes('-zoomed', True)
    root.title("Click the center 0.01")

    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    for cls in classes:
        if(cls not in ['Container_R', 'unspecified', 'nothing']):
            currentClassId = classes[cls]
            ttk.Button(mainframe, text=cls, command=lambda currentClassId=currentClassId: selectObjectType(currentClassId)).grid(column=classes[cls], row=1, sticky=E)

    imageLabel = ttk.Label(mainframe, cursor = "crosshair")
    nextImage()
    imageLabel.grid(column=0, row=2, columnspan=12, sticky=(W, E))

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    root.bind('<Return>', saveAndNext)
    root.bind('<space>', saveAndNext)
    root.bind('<Escape>', quitprog)
    root.bind("<Delete>", deleteImage)
    root.bind("<Right>", nextImage)
    imageLabel.bind("<Button-1>", onclick)
    imageLabel.bind("<Button-2>", undo)


    root.focus_set()

    root.mainloop()


if __name__ == '__main__':
    main()
