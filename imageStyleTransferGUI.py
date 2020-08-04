from tkinter import filedialog
import tkinter as tk
import imageStyleTransfer as ist

styleImage = ""

def loadStyle(event = None):
    global styleImage
    styleImage = filedialog.askopenfilename()
    print(styleImage)

def main():
    root = tk.Tk()
    root.geometry("200x150")
    frame = tk.Frame(root)
    frame.pack()

    leftframe = tk.Frame(root)
    leftframe.pack(side="left")

    rightframe = tk.Frame(root)
    rightframe.pack(side="right")

    label = tk.Label(frame, text = "Image Style Transfer")
    label.pack()

    loadStyleButton = tk.Button(leftframe, text = "Load Style Image",
                                height = 1, width = 14,
                                command = loadStyle)
    loadStyleButton.pack()

    loadContentButton = tk.Button(leftframe, text = "Load Content Image", height = 1, width = 14)
    loadContentButton.pack()

    # Displays
    styleText = tk.Text(rightframe, height = 1, width = 14)
    styleText.pack()
    styleText.insert(tk.END, styleImage)
    
    root.title("TF Image Style Transfer")
    root.mainloop()

    
if __name__ == "__main__":
    main()
