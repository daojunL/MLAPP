from tkinter import *
from Step1 import Step1


def main():
    root = Tk()
    root.title("ML APP")
    root.resizable(width=False, height=False)

    step1 = Step1(root)
    step1.pack_frame()

    root.mainloop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


