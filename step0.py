from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
import pandas as pd


global data_set


def load_data(file_path, str_separator, int_header, int_duplicates):
    global data_set

    if len(step1_frame.pack_slaves()) > 3:
        step1_frame.pack_slaves()[-1].destroy()

    try:
        root.geometry('700x440')
        if int_header == 1:
            int_header = 0
        else:
            int_header = None
        date_set = pd.read_table(file_path, sep=str_separator, header=int_header)

        if int_duplicates == 1:
            date_set = date_set.drop_duplicates()

        # Getting column names and its width
        columns, col_width = list(date_set.columns.astype('str')), []
        date_set.columns = columns
        for col in columns:
            width = date_set.head()[col].astype('str').str.len().max()
            if width >= len(col):
                col_width.append(width * 6)
            else:
                col_width.append(len(col) * 6 + 25)

        data_table = ttk.Treeview(wrapper3, columns=tuple(columns), show="headings", height=5)

        rows = date_set.head().values
        for row in rows:
            data_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            data_table.heading(i, text=col, anchor=W)
            data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

        hsb = ttk.Scrollbar(wrapper3, orient="horizontal", command=data_table.xview)
        data_table.configure(xscrollcommand=hsb.set)
        success_label = Label(step1_frame, text='Data read successfully',
                              font=tkfont.Font(family="Times New Roman", size=12))

        if len(wrapper3.pack_slaves()) == 1:
            wrapper3.pack_slaves()[-1].destroy()
        elif len(wrapper3.pack_slaves()) == 2:
            wrapper3.pack_slaves()[-1].destroy()
            wrapper3.pack_slaves()[-1].destroy()

        data_table.pack(padx=5, pady=5)
        hsb.pack(fill='x')
        success_label.pack(side=LEFT, anchor=W, padx=20, pady=5)

    except Exception:
        root.geometry('700x300')
        if len(wrapper3.pack_slaves()) == 1:
            wrapper3.pack_slaves()[-1].destroy()
        elif len(wrapper3.pack_slaves()) == 2:
            wrapper3.pack_slaves()[-1].destroy()
            wrapper3.pack_slaves()[-1].destroy()

        data_label = Label(wrapper3, text='No data found', font=tkfont.Font(family="Times New Roman", size=12))
        data_label.pack(anchor=W, padx=3)
        error_label = Label(step1_frame, text='Error: please provide a valid file path or delimiter',
                            font=tkfont.Font(family="Times New Roman", size=12), fg='red')
        error_label.pack(side=LEFT, anchor=W, padx=20, pady=5)


root = Tk()
root.title("ML APP")
root.geometry('700x275')

step1_frame = Frame(root)

# First section of the page
wrapper1 = LabelFrame(step1_frame, text='Welcome to the ML APP', font=tkfont.Font(family="Times New Roman", size=16))

description = 'We are here to assist you explore and analyze your data. Our Machine Learning models will help you out.'
description_label = Label(wrapper1, text=description, font=tkfont.Font(family="Times New Roman", size=12))
project_label = Label(wrapper1, text='What is your project name?', font=tkfont.Font(family="Times New Roman", size=12))
project_entry = Entry(wrapper1, width=50)

# Packing of elements into the section
description_label.pack(anchor=W, padx=3)
project_label.pack(side=LEFT, anchor=W, padx=3, pady=10)
project_entry.pack(side=LEFT, padx=3)

# Second section of the page
wrapper2 = LabelFrame(step1_frame, text='Step 1: Let\'s import your data:', font=tkfont.Font(family="Times New Roman", size=14))

frame1 = Frame(wrapper2)
file_path_label = Label(frame1, text='Filepath:', font=tkfont.Font(family="Times New Roman", size=12))
file_path_entry = Entry(frame1, width=40)
delimiter_label = Label(frame1, text='Delimiter: ', font=tkfont.Font(family="Times New Roman", size=12))
delimiter_combo = ttk.Combobox(frame1, width=3)
delimiter_combo.state(["readonly"])
delimiter_combo["values"] = [",", ";", ":", "|", "\\t"]
delimiter_combo.set(",")

option_label = Label(wrapper2, text='Options:', font=tkfont.Font(family="Times New Roman", size=12))
option1_var, option2_var = IntVar(), IntVar()
option_button1 = Checkbutton(wrapper2, text="Data has headers", font=tkfont.Font(family="Times New Roman", size=10),
                             variable=option1_var)
option_button2 = Checkbutton(wrapper2, text="Remove duplicates", font=tkfont.Font(family="Times New Roman", size=10),
                             variable=option2_var)
load_button = Button(wrapper2, text='Load Data', font=tkfont.Font(family="Times New Roman", size=10),
                     command=lambda: load_data(file_path_entry.get(), delimiter_combo.get(), option1_var.get(), option2_var.get()))

# Packing of elements into the section
file_path_label.pack(side=LEFT, padx=3, pady=5)
file_path_entry.pack(side=LEFT, padx=3)
delimiter_label.pack(side=LEFT, padx=3)
delimiter_combo.pack(side=LEFT)

frame1.pack(side=TOP, anchor=W)
option_label.pack(side=LEFT, padx=2, pady=5)
option_button1.pack(side=LEFT)
option_button2.pack(side=LEFT)
load_button.pack(side=RIGHT, padx=65)

# Third section of the page
wrapper3 = LabelFrame(step1_frame, text='Experiment dataset: ', font=tkfont.Font(family="Times New Roman", size=14))

data_label = Label(wrapper3, text='No data found', font=tkfont.Font(family="Times New Roman", size=12))
data_label.pack(anchor=W, padx=3)

wrapper1.pack(fill='x', expand='no', padx=20, pady=5)
wrapper2.pack(fill='x', expand='no', padx=20, pady=5)
wrapper3.pack(fill='both', expand='yes', padx=20, pady=5)

step1_frame.pack(fill='both')

root.mainloop()
