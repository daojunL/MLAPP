from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from ModelSummarizer import ModelSummarizer


class ModelTrainer:
    def __init__(self, root, data_set, project_title, previous_frame):
        self.__root = root
        self.__root.geometry('1000x550')
        self.__label_font = tkfont.Font(family="Times New Roman", size=12)
        self.__button_font = tkfont.Font(family="Times New Roman", size=10)
        self.__title_font = tkfont.Font(family="Times New Roman", size=14)
        self.__data_set = data_set

        self.__project_title = project_title
        self.__previous_frame = previous_frame

        self.__step4_frame = Frame(self.__root)

        # Title of the page
        project_title_label = Label(self.__step4_frame, text="Project name: " + self.__project_title,
                                    font=tkfont.Font(family="Times New Roman", size=16))

        # First section of the page
        self.__wrapper1 = LabelFrame(self.__step4_frame, text='Dataset Head (5 rows): ', font=self.__title_font)
        columns, col_width = list(self.__data_set.columns), []
        for col in columns:
            width = self.__data_set.head()[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 25)

        self.__data_table = ttk.Treeview(self.__wrapper1, columns=tuple(columns), show="headings", height=7)

        rows = self.__data_set.head(5).values
        for row in rows:
            self.__data_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__data_table.heading(i, text=col, anchor=W)
            self.__data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

        vsb2 = ttk.Scrollbar(self.__wrapper1, orient="vertical", command=self.__data_table.yview)
        hsb2 = ttk.Scrollbar(self.__wrapper1, orient="horizontal", command=self.__data_table.xview)
        self.__data_table.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)

        hsb2.pack(side=BOTTOM, fill='x', padx=5, pady=5)
        vsb2.pack(side=RIGHT, fill='y')
        self.__data_table.pack(side=TOP, fill='y', expand='yes', padx=5, anchor=W)

        # Second section of the page
        self.__wrapper2 = LabelFrame(self.__step4_frame, text='Step 4: Model Setup', font=self.__title_font)
        self.__subframe1 = Frame(self.__wrapper2, width=1000, height = 105)
        self.__subframe1.pack(fill='x', expand='no', padx=10, pady=5)
        self.__sub_subframe1 = Frame(self.__subframe1)
        self.__sub_subframe1.pack(fill="both", expand = "yes", padx=10, pady=2)
        self.__sub_subframe2 = Frame(self.__subframe1)
        self.__sub_subframe2.pack(fill="both", expand = "yes", padx=10, pady=2)
        self.__canvas = Canvas(self.__sub_subframe1)
        self.__canvas2 = Canvas(self.__sub_subframe1)
        self.__canvas.pack(side=TOP, expand="yes", fill="x")
        self.__canvas2.pack(side=BOTTOM, expand="yes", fill="x")
        self.__canvas.configure(width=950, height=50)
        self.__canvas2.configure(width=950, height=50)
        self.__canvas_frame=Frame(self.__canvas)
        self.__canvas_frame2=Frame(self.__canvas2)
        self.__canvas.create_window((0,0), window=self.__canvas_frame, anchor="nw")
        self.__canvas2.create_window((0,0), window=self.__canvas_frame2, anchor="nw")

        xScrollBar = ttk.Scrollbar(self.__canvas_frame, orient="horizontal", command=self.__canvas.xview)
        xScrollBar.pack(side=BOTTOM, fill="x")
        self.__canvas.configure(xscrollcommand=xScrollBar.set)
        self.__canvas.bind('<Configure>', lambda e: self.__canvas.configure(scrollregion = self.__canvas.bbox('all')))

        xScrollBar2 = ttk.Scrollbar(self.__canvas_frame2, orient="horizontal", command=self.__canvas2.xview)
        xScrollBar2.pack(side=BOTTOM, fill="x")
        self.__canvas2.configure(xscrollcommand=xScrollBar2.set)
        self.__canvas2.bind('<Configure>', lambda e: self.__canvas2.configure(scrollregion = self.__canvas2.bbox('all')))

        column_names = list(self.__data_set.columns.values)
        num_variables = len(column_names)
        self.__check_button_control_list = []
        check_button_list = []

        for i in range(num_variables):
            control_variable = IntVar()
            control_variable.set(0)
            self.__check_button_control_list.append(control_variable)

        for i in range(num_variables):
            check_button = ttk.Checkbutton(self.__canvas_frame, variable=self.__check_button_control_list[i], text=column_names[i])
            check_button_list.append(check_button)

        select_features_label = Label(self.__canvas_frame, text='Select features: ', font=self.__label_font)
        select_features_label.pack(side=LEFT, padx=5, pady=2)

        for i in range(num_variables):
            # data_type = self.__data_set.dtypes[i]
            # if data_type == "int64" or data_type == "float64":
            check_button_list[i].pack(side=LEFT, padx=5, pady=2)

        select_y_label = Label(self.__canvas_frame2, text='Select label: ', font=self.__label_font)
        select_y_label.pack(side=LEFT, padx=5, pady=2)
        self.__control_label_var = IntVar()
        self.__control_label_var.set(0)

        for i in range(num_variables):
            label_radio_button = Radiobutton(self.__canvas_frame2, value=i+1, variable=self.__control_label_var, text=column_names[i])
            data_type = self.__data_set.dtypes[i]
            if data_type == "int64" or data_type == "float64":
                label_radio_button.pack(side=LEFT, padx=5, pady=2)

        select_ratio_label = Label(self.__sub_subframe2, text='Select train/test ratio: ', font=self.__label_font)
        select_ratio_label.grid(row=2, column=0, sticky=W, pady=1)
        ratio = ["1", "2", "3", "4", "5"]
        self.__control_ratio_var = IntVar()
        self.__control_ratio_var.set(0)
        ratio_radio_button1 = Radiobutton(self.__sub_subframe2, value=1, variable=self.__control_ratio_var, text=ratio[0])
        ratio_radio_button2 = Radiobutton(self.__sub_subframe2, value=2, variable=self.__control_ratio_var, text=ratio[1])
        ratio_radio_button3 = Radiobutton(self.__sub_subframe2, value=3, variable=self.__control_ratio_var, text=ratio[2])
        ratio_radio_button4 = Radiobutton(self.__sub_subframe2, value=4, variable=self.__control_ratio_var, text=ratio[3])
        ratio_radio_button5 = Radiobutton(self.__sub_subframe2, value=5, variable=self.__control_ratio_var, text=ratio[4])
        ratio_radio_button1.grid(row=2, column=1)
        ratio_radio_button2.grid(row=2, column=2)
        ratio_radio_button3.grid(row=2, column=3, padx = 50)
        ratio_radio_button4.grid(row=2, column=4, padx = 50)
        ratio_radio_button5.grid(row=2, column=5)

        select_model_label = Label(self.__sub_subframe2, text='Select model: ', font=self.__label_font)
        select_model_label.grid(row=3, column=0, sticky=W, pady=1)
        self.__control_model_var = IntVar()
        self.__control_model_var.set(0)
        model = ["Linear Regression", "Decision Tree Regression", "SVM"]
        model_radio_button1 = Radiobutton(self.__sub_subframe2, value=1, variable=self.__control_model_var, text=model[0])
        model_radio_button2 = Radiobutton(self.__sub_subframe2, value=2, variable=self.__control_model_var, text=model[1])
        model_radio_button3 = Radiobutton(self.__sub_subframe2, value=3, variable=self.__control_model_var, text=model[2])
        model_radio_button1.grid(row=3, column=1)
        model_radio_button2.grid(row=3, column=2)
        model_radio_button3.grid(row=3, column=3)

        submit_button = Button(self.__sub_subframe2, text='Confirm Model', font=self.__button_font, width=14,
                                    command=self.confirm_model)
        submit_button.grid(row=3, column=5)

        # Navigation buttons
        self.__next_button = Button(self.__step4_frame, text='Next step', font=self.__button_font, width=14,
                                    command=self.unpack_frame_forward)
        self.__previous_button = Button(self.__step4_frame, text='Previous step', font=self.__button_font, width=14,
                                        command=self.unpack_frame_previous)

        # Packing all elements of the step4_frame
        project_title_label.pack(fill='x', pady=3)
        self.__wrapper1.pack(fill='x', expand='no', padx=10, pady=2)
        self.__wrapper2.pack(fill='x', expand='no', padx=10, pady=2)

        self.__previous_button.pack(side=LEFT, padx=10, pady=5)
        self.__next_button.pack(side=RIGHT, padx=10, pady=5)

    def confirm_model(self):
        try:
            self.__subframe2.pack_forget()
        except:
            print()
        self.__subframe2 = Frame(self.__wrapper2)
        self.__subframe2.pack(fill='x', expand='no', padx=20, pady=5)
        model_var = int(self.__control_model_var.get())  # get which model
        ratio_var = int(self.__control_ratio_var.get())  # get train/test ratio
        label_var = int(self.__control_label_var.get())  # get the label variable
        colnames = self.__data_set.columns.values.tolist()
        features = []
        for i in range(len(self.__check_button_control_list)):
            if int(self.__check_button_control_list[i].get()) == 1:
                features.append(colnames[i])

        x = self.__data_set.loc[:, features]
        for i in range(len(features)):
            feature_data_type = x.dtypes[i]
            if feature_data_type != "int64" and feature_data_type != "float64":
                x[features[i]] = pd.factorize(x[features[i]])[0]

        y = self.__data_set.loc[:, colnames[label_var - 1]]
        test_size = 1 / (1 + ratio_var)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=123)
        self.__x_train = x_train
        self.__x_test = x_test
        self.__y_train = y_train
        self.__y_test = y_test
        if model_var == 1:  # linear regression model
            lr_model = LinearRegression()
            lr_model.fit(self.__x_train, self.__y_train)
            self.__model = lr_model
            success_message = Label(self.__subframe2, text="Model has been successfully set up ",
                                    font=self.__label_font)
            success_message.pack(side=BOTTOM)
        elif model_var == 2:  # Decision Tree Model
            max_depth_label = Label(self.__subframe2, text="Enter the maximum depth", font=self.__label_font)
            max_depth_label.pack(side=LEFT, padx=5, pady=2)
            self.__max_depth_entry = Entry(self.__subframe2, width=30)
            self.__max_depth_entry.pack(side=LEFT, padx=5, pady=2)
            confirm_button = Button(self.__subframe2, text='confirm', font=self.__button_font, width=14,
                                    command=self.confirm_tree_model)
            confirm_button.pack(side=LEFT, padx=5, pady=2)

        elif model_var == 3:  # svm model
            svm_model = svm.SVR()
            svm_model.fit(self.__x_train, self.__y_train)
            self.__model = svm_model
            success_message = Label(self.__subframe2, text="Model has been successfully set up ",
                                    font=self.__label_font)
            success_message.pack(side=BOTTOM)

    def confirm_tree_model(self):
        max_depth = self.__max_depth_entry.get()
        tree_model = DecisionTreeRegressor(max_depth=int(max_depth))
        tree_model.fit(self.__x_train, self.__y_train)
        self.__model = tree_model
        success_message = Label(self.__subframe2, text="Model has been successfully set up ",font=self.__label_font)
        success_message.pack(side=BOTTOM)

    def pack_frame(self):
        self.__step4_frame.pack(fill='both')

    def unpack_frame_forward(self):
        # Unpack the model trainer main frame
        self.__step4_frame.pack_forget()
        ModelSummarizer(self.__root, self.__x_test,  self.__y_test, self.__model, self.__project_title, self.__step4_frame).pack_frame()

    def unpack_frame_previous(self):
        self.__step4_frame.pack_forget()
        self.__root.geometry('1200x793')
        self.__previous_frame.pack(fill='both')