import tkinter as tk

def main():
    root = tk.Tk()
    root.geometry("200x150")
    frame = tk.Frame(root)
    frame.pack()

    leftframe = tk.Frame(root)
    leftframe.pack(side="left")

    rightframe = tk.Frame(root)
    rightframe.pack(side="right")

    label = tk.Label(frame, text = "Hello world")
    label.pack()

    
    root.title("Test")
    root.mainloop()

if __name__ == "__main__":
    main()
