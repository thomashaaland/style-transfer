from tkinter import filedialog
import tkinter as tk
import imageStyleTransfer as ist
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Make the root
    root = tk.Tk()
    root.geometry("1400x500")

    label = tk.Label(root, text = "Image Style Transfer")
    label.grid(column = 0, row = 0)

    # Prep for displaying style image
    canvasStyle = tk.Canvas(root, width = 300, height = 300)
    canvasStyle.grid(column = 2, row = 3)

    # Prep for the style image
    styleImageName = ""
    styleImage = None
    styleText = tk.Label(root, height = 1, width = 14, text = styleImageName)
    styleText.grid(column = 1, row = 1)
    
    def loadStyle(event = None):
        global styleImageName
        global styleImage
        styleImageName = filedialog.askopenfilename()
        styleText.configure(text = styleImageName)
        styleImage = ImageTk.PhotoImage(Image.open(styleImageName).resize((300, 300), Image.ANTIALIAS))
        canvasStyle.create_image(150, 150, image=styleImage)
        canvasStyle.image = styleImage
        
    loadStyleButton = tk.Button(root, text = "Load Style Image",
                                height = 1, width = 14,
                                command = loadStyle)
    loadStyleButton.grid(column = 0, row = 1)

    # Prep for displaying content image
    canvasContent = tk.Canvas(root, width = 300, height = 300)
    canvasContent.grid(column = 4, row = 3)
    
    # Prep for content image
    contentImageName = ""
    contentImage = None
    contentText = tk.Label(root, height = 1, width = 14, text = contentImageName)
    contentText.grid(column = 1, row = 2)

    def loadContent(event = None):
        global contentImageName
        global contentImage
        contentImageName = filedialog.askopenfilename()
        contentText.configure(text = contentImageName)
        contentImage = ImageTk.PhotoImage(Image.open(contentImageName).resize((300, 300), Image.ANTIALIAS))
        canvasContent.create_image(150,150, image=contentImage)
        canvasContent.image = contentImage
        
    loadContentButton = tk.Button(root, text = "Load Content Image",
                                  height = 1, width = 14,
                                  command = loadContent)
    loadContentButton.grid(column = 0, row = 2)

    # Prep for displaying result
    canvasResult = tk.Canvas(root, width = 300, height = 300)
    canvasResult.grid(column = 6, row = 3)
    
    # Prep for result button
    resultImage = None
    
    def generateResult(event = None):
        global contentImageName, styleImageName, resultImage

        resultImage = None
        rawResult = np.array(ist.toImage(ist.makeImg(contentImageName, styleImageName)))
        rawResult = Image.fromarray(np.uint8(rawResult * 255))
        resultImage = ImageTk.PhotoImage(rawResult.resize((300, 300), Image.ANTIALIAS))
        canvasResult.create_image(150,150, image = resultImage)
        canvasResult.image = resultImage

    generateResultButton = tk.Button(root, text = "Generate result",
                                     height = 1, width = 14,
                                     command = generateResult)
    generateResultButton.grid(column = 6, row = 2)

    # Display everything
    root.title("TF Image Style Transfer")
    root.mainloop()

    
if __name__ == "__main__":
    main()
