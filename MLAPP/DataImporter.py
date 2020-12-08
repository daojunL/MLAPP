from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
import pandas as pd
from DataModifier import DataModifier


# Class in charge of importing the data given the user inputs
class DataImporter:
    def __init__(self, root):
        self.__root = root
        self.__root.geometry('700x275')

        # Font objects to be used on the section
        self.__label_font = tkfont.Font(family="Times New Roman", size=12)
        self.__button_font = tkfont.Font(family="Times New Roman", size=10)
        self.__title_font = tkfont.Font(family="Times New Roman", size=14)

        # Data frame that will contain the loaded data
        self.__data_set = pd.DataFrame()
        self.__project_title = 'Generic project'

        # Main frame of the Data importer (Step 1)
        self.__step1_frame = Frame(self.__root)

        # First section of the page - Project description and name
        self.__wrapper1 = LabelFrame(self.__step1_frame, text='Welcome to the ML APP',
                                     font=tkfont.Font(family="Times New Roman", size=16))

        description = 'We are here to assist you explore and analyze your data. ' + \
                      'Our Machine Learning models will help you out.'
        description_label = Label(self.__wrapper1, text=description, font=self.__label_font)
        project_label = Label(self.__wrapper1, text='What is your project name?', font=self.__label_font)
        self.__project_entry = Entry(self.__wrapper1, width=50)

        # Packing of elements into the first section
        description_label.pack(anchor=W, padx=3)
        project_label.pack(side=LEFT, anchor=W, padx=3, pady=10)
        self.__project_entry.pack(side=LEFT, padx=3)

        # Second section of the page - Data set importer
        self.__wrapper2 = LabelFrame(self.__step1_frame, text='Step 1: Let\'s import the data:', font=self.__title_font)

        frame1 = Frame(self.__wrapper2)
        file_path_label = Label(frame1, text='Filepath:', font=self.__label_font)
        self.file_path_entry = Entry(frame1, width=75)
        self.file_path_entry.bind("<KeyRelease-Return>", self.load_data)

        delimiter_label = Label(frame1, text='Delimiter: ', font=self.__label_font)
        self.delimiter_combo = ttk.Combobox(frame1, width=3)
        self.delimiter_combo.state(["readonly"])
        self.delimiter_combo["values"] = [",", ";", ":", "|", "\\t"]
        self.delimiter_combo.set(",")

        # Importing options like header column and duplicate removal
        option_label = Label(self.__wrapper2, text='Options:', font=self.__label_font)
        self.option1_var, self.option2_var = IntVar(), IntVar()
        option_button1 = Checkbutton(self.__wrapper2, text="Data has headers",
                                     font=self.__button_font, variable=self.option1_var)
        option_button2 = Checkbutton(self.__wrapper2, text="Remove duplicates",
                                     font=self.__button_font, variable=self.option2_var)
        load_button = Button(self.__wrapper2, text='Load Data', font=self.__button_font, width=14,
                             command=self.load_data)

        # Packing of elements into the second section
        file_path_label.pack(side=LEFT, padx=3, pady=5)
        self.file_path_entry.pack(side=LEFT, padx=3)
        delimiter_label.pack(side=LEFT, padx=3)
        self.delimiter_combo.pack(side=LEFT)

        frame1.pack(side=TOP, anchor=W)
        option_label.pack(side=LEFT, padx=2, pady=5)
        option_button1.pack(side=LEFT)
        option_button2.pack(side=LEFT)
        load_button.pack(side=RIGHT, padx=21)

        # Third section of the page - Data set row sample
        self.__wrapper3 = LabelFrame(self.__step1_frame, text='Experiment dataset: ', font=self.__title_font)

        data_label = Label(self.__wrapper3, text='No data found', font=self.__label_font)

        # Packing of elements into the section
        data_label.pack(anchor=W, padx=3)

        # Packing of elements into step1_frame
        self.__wrapper1.pack(fill='x', expand='no', padx=20, pady=5)
        self.__wrapper2.pack(fill='x', expand='no', padx=20, pady=5)
        self.__wrapper3.pack(fill='both', expand='yes', padx=20, pady=5)

    def load_data(self, event=None):
        # Reset data set
        self.__data_set = pd.DataFrame()

        # Get importing set up variables
        file_path = self.file_path_entry.get()
        str_separator = self.delimiter_combo.get()
        int_header = self.option1_var.get()
        int_duplicates = self.option2_var.get()

        # Check if a title is in the project name entry box
        if len(str(self.__project_entry.get())) > 0:
            self.__project_title = str(self.__project_entry.get())

        # Check if a success/error message is packed
        if len(self.__step1_frame.pack_slaves()) == 4:
            self.__step1_frame.pack_slaves()[-1].destroy()
        if len(self.__step1_frame.pack_slaves()) == 5:
            self.__step1_frame.pack_slaves()[-1].destroy()
            self.__step1_frame.pack_slaves()[-1].destroy()

        try:
            self.__root.geometry('700x440')

            # Check if the header option was selected
            if int_header == 1:
                int_header = 0
            else:
                int_header = None

            # Load the file on the data set variable
            self.__data_set = pd.read_table(file_path, sep=str_separator, header=int_header)

            # Check if the remove duplicate option was selected
            if int_duplicates == 1:
                self.__data_set = self.__data_set.drop_duplicates()

            # Getting column names and its width
            columns, col_width = list(self.__data_set.columns), []
            for col in columns:
                width = self.__data_set.head()[col].astype('str').str.len().max()
                if width >= len(str(col)):
                    col_width.append(width * 6 + 15)
                else:
                    col_width.append(len(str(col)) * 6 + 25)

            # Create table widget to show the first 5 rows of the data set
            data_table = ttk.Treeview(self.__wrapper3, columns=tuple(columns), show="headings", height=5)

            rows = self.__data_set.head().values
            for row in rows:
                data_table.insert('', 'end', values=tuple(row))

            for i, col in enumerate(columns):
                data_table.heading(i, text=col, anchor=W)
                data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

            hsb = ttk.Scrollbar(self.__wrapper3, orient="horizontal", command=data_table.xview)
            data_table.configure(xscrollcommand=hsb.set)

            rows, columns = self.__data_set.shape[0], self.__data_set.shape[1]

            # Create success message
            text = '*' + str(columns) + ' columns with ' + str(rows) + ' rows were loaded successfully'
            success_label = Label(self.__step1_frame, text=text, font=self.__label_font)

            next_button = Button(self.__step1_frame, text='Next step', font=self.__button_font,
                                 command=self.unpack_frame, width=14)

            for element in self.__wrapper3.pack_slaves():
                element.destroy()

            # Packing of elements into the third section
            data_table.pack(padx=5, anchor=W)
            hsb.pack(fill='x', padx=5, pady=5)
            success_label.pack(side=LEFT, anchor=W, padx=20, pady=5)
            next_button.pack(side=RIGHT, padx=45)

        except Exception:
            self.__root.geometry('700x300')
            for element in self.__wrapper3.pack_slaves():
                element.destroy()

            # Create error message
            data_label = Label(self.__wrapper3, text='No data found', font=self.__label_font)
            data_label.pack(anchor=W, padx=3)
            error_label = Label(self.__step1_frame, text='Error: please provide a valid file path or delimiter',
                                font=self.__label_font, fg='red')
            error_label.pack(side=LEFT, anchor=W, padx=20, pady=5)

    def pack_frame(self):
        # Pack the data importer main frame
        self.__step1_frame.pack(fill='both')

    def unpack_frame(self):
        # Check if a title is in the project name entry box
        if len(str(self.__project_entry.get())) > 0:
            self.__project_title = str(self.__project_entry.get())

        # Unpack the data importer main frame
        self.__step1_frame.pack_forget()

        # Call the next step class (Data modifier)
        DataModifier(self.__root, self.__data_set, self.__project_title,  self.__step1_frame).pack_frame()
