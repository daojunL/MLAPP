from tkinter import *
from DataImporter import DataImporter


def main():
    # Creation of the root frame of the application
    root = Tk()
    root.title("ML APP")
    root.resizable(width=False, height=False)

    # Call of the first step window (Data importer)
    step1 = DataImporter(root)
    step1.pack_frame()

    root.mainloop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


