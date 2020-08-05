from tkinter import filedialog
import tkinter as tk
import imageStyleTransfer as ist

def main():
    root = tk.Tk()
    root.geometry("350x150")
    #frame = tk.Frame(root)
    #frame.pack()

    label = tk.Label(root, text = "Image Style Transfer")
    label.grid(column = 0, row = 0)

    # Style loading
    styleImage = ""
    styleText = tk.Label(root, height = 1, width = 14, text = styleImage)
    styleText.grid(column = 1, row = 1)
    
    def loadStyle(event = None):
        styleImage = filedialog.askopenfilename()
        styleText.configure(text = styleImage)

    loadStyleButton = tk.Button(root, text = "Load Style Image",
                                height = 1, width = 14,
                                command = loadStyle)
    loadStyleButton.grid(column = 0, row = 1)

    # Content loading
    contentImage = ""
    contentText = tk.Label(root, height = 1, width = 14, text = contentImage)
    contentText.grid(column = 1, row = 2)

    def loadContent(event = None):
        contentImage = filedialog.askopenfilename()
        contentText.configure(text = contentImage)
        
    loadContentButton = tk.Button(root, text = "Load Content Image",
                                  height = 1, width = 14,
                                  command = loadContent)
    loadContentButton.grid(column = 0, row = 2)
    
    root.title("TF Image Style Transfer")
    root.mainloop()

    
if __name__ == "__main__":
    main()
