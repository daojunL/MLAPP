from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
from Step3 import Step3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Step2:
    def __init__(self, root, data_set, project_title, previous_frame):
        self.__root = root
        self.__root.geometry('800x720')
        self.__label_font = tkfont.Font(family="Times New Roman", size=12)
        self.__button_font = tkfont.Font(family="Times New Roman", size=10)
        self.__title_font = tkfont.Font(family="Times New Roman", size=14)
        self.__data_set = data_set
        self.__project_title = project_title
        self.__previous_frame = previous_frame
        self.__column_types = pd.DataFrame()
        self.__vect_columns = {}

        self.__step2_frame = Frame(self.__root)

        # Title of the page
        project_title_label = Label(self.__step2_frame, text="Project name: " + self.__project_title,
                                    font=tkfont.Font(family="Times New Roman", size=16))

        # First section of the page
        self.__wrapper1 = LabelFrame(self.__step2_frame, text='Step 2: Let\'s process your columns:',
                                     font=self.__title_font)
        self.__wrapper1.columnconfigure(0, weight=1, uniform="group")
        self.__wrapper1.columnconfigure(1, weight=1, uniform="group")

        table_frame = Frame(self.__wrapper1)

        # Filling table with column names and types
        self.generate_table_type()
        columns = list(self.__column_types.columns)

        col_width = []
        for col in columns:
            width = self.__column_types[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 10)

        self.__columns_table = ttk.Treeview(table_frame, columns=tuple(columns), show="headings", height=7)

        rows = self.__column_types.values
        for row in rows:
            self.__columns_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__columns_table.heading(i, text=col, anchor=W)
            self.__columns_table.column(col, width=col_width[i], stretch=NO, anchor=W)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.__columns_table.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.__columns_table.xview)
        self.__columns_table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        hsb.pack(side=BOTTOM, fill='x', padx=5)
        vsb.pack(side=RIGHT, fill='y')
        self.__columns_table.pack(side=TOP, fill='both', expand='yes', anchor=W)

        table_frame.grid(row=0, column=0, padx=5, sticky=W)

        # Creating metadata options
        self.__metadata_frame = Frame(self.__wrapper1)
        columns = list(self.__column_types['Column'])
        types = list(self.__column_types['Type'])

        column_label = Label(self.__metadata_frame, text='Select column:', font=self.__label_font)
        self.__column_combo = ttk.Combobox(self.__metadata_frame, width=20, height=5)
        self.__column_combo.state(["readonly"])
        self.__column_combo["values"] = columns
        self.__column_combo.set(columns[0])
        self.__column_combo.bind('<<ComboboxSelected>>', self.combo_selection)

        name_label = Label(self.__metadata_frame, text='Column name:', font=self.__label_font)
        self.__name_entry = Entry(self.__metadata_frame, width=25)
        self.__name_control = StringVar()
        self.__name_control.set(self.__column_combo.get())
        self.__name_entry["textvariable"] = self.__name_control

        type_label = Label(self.__metadata_frame, text='Column type:', font=self.__label_font)
        self.__type_combo = ttk.Combobox(self.__metadata_frame, width=22)
        self.__type_combo.state(["readonly"])
        self.__type_combo["values"] = ['object', 'int64', 'float64', 'category', 'bool']
        self.__type_combo.set(types[0])

        self.__drop_option_var = IntVar()
        drop_option_checkbutton = Checkbutton(self.__metadata_frame, text="Drop column?",
                                              font=self.__button_font, variable=self.__drop_option_var)

        change_button = Button(self.__metadata_frame, text='Change', font=self.__button_font, width=7,
                               command=self.metadata_click)

        column_label.grid(row=0, column=0, sticky=W, pady=3)
        self.__column_combo.grid(row=0, column=1, padx=10, sticky=W)
        name_label.grid(row=1, column=0, sticky=W, pady=3)
        self.__name_entry.grid(row=1, column=1, padx=10, sticky=W)
        type_label.grid(row=2, column=0, sticky=W, pady=3)
        self.__type_combo.grid(row=2, column=1, padx=10, sticky=W)
        drop_option_checkbutton.grid(row=3, column=0, sticky=W, pady=3)
        change_button.grid(row=4, column=2, sticky=W)

        self.__metadata_frame.grid(row=0, column=1, sticky=N + S + W + E)

        # Second section of the page
        self.__wrapper2 = Frame(self.__step2_frame)
        self.__wrapper2.columnconfigure(0, weight=1, uniform="group")
        self.__wrapper2.columnconfigure(1, weight=1, uniform="group")

        # Numerical column transformations
        self.__num_col_frame = LabelFrame(self.__wrapper2, text='Numerical transformations:', font=self.__title_font)

        num_column_label = Label(self.__num_col_frame, text='Select column:', font=self.__label_font)
        self.__num_column_combo = ttk.Combobox(self.__num_col_frame, width=20, height=5)
        self.__num_column_combo.state(["readonly"])
        self.fill_numeric_combobox()

        num_null_label = Label(self.__num_col_frame, text='1-Null values:', font=self.__label_font)
        self.__num_null_control = IntVar()
        self.__num_null_control.set(1)
        num_null_button1 = ttk.Radiobutton(self.__num_col_frame, value=1, variable=self.__num_null_control,
                                           text="Drop row")
        num_null_button2 = ttk.Radiobutton(self.__num_col_frame, value=2, variable=self.__num_null_control,
                                           text="Fill mean")
        num_null_button3 = ttk.Radiobutton(self.__num_col_frame, value=3, variable=self.__num_null_control,
                                           text="Fill median")

        num_scaling_label = Label(self.__num_col_frame, text='3-Scaler:', font=self.__label_font)
        self.__num_scaling_control = IntVar()
        self.__num_scaling_control.set(3)
        num_scaling_button1 = ttk.Radiobutton(self.__num_col_frame, value=1, variable=self.__num_scaling_control,
                                              text="Standard")
        num_scaling_button2 = ttk.Radiobutton(self.__num_col_frame, value=2, variable=self.__num_scaling_control,
                                              text="MinMax")
        num_scaling_button3 = ttk.Radiobutton(self.__num_col_frame, value=3, variable=self.__num_scaling_control,
                                              text="None")

        num_trans_label = Label(self.__num_col_frame, text='2-Transform:', font=self.__label_font)
        self.__num_trans_control = IntVar()
        self.__num_trans_control.set(3)
        num_trans_button1 = ttk.Radiobutton(self.__num_col_frame, value=1, variable=self.__num_trans_control,
                                            text="Sqrt")
        num_trans_button2 = ttk.Radiobutton(self.__num_col_frame, value=2, variable=self.__num_trans_control,
                                            text="Log")
        num_trans_button3 = ttk.Radiobutton(self.__num_col_frame, value=3, variable=self.__num_trans_control,
                                            text="None")

        self.__new_num_option_var = IntVar()
        num_new_checkbutton = Checkbutton(self.__num_col_frame, text="Create new column?",
                                          font=self.__button_font, variable=self.__new_num_option_var)

        num_apply_button = Button(self.__num_col_frame, text='Apply', font=self.__button_font, width=7,
                                  command=self.transform_num_col_click)

        # Packing elements into numerical column section
        num_column_label.grid(row=0, column=0, sticky=W, pady=1)
        self.__num_column_combo.grid(row=0, column=1, padx=5, sticky=W, columnspan=2)
        num_null_label.grid(row=1, column=0, sticky=W, pady=1)
        num_null_button1.grid(row=1, column=1, sticky=W)
        num_null_button2.grid(row=1, column=2, sticky=W)
        num_null_button3.grid(row=1, column=3, sticky=W)
        num_trans_label.grid(row=2, column=0, sticky=W, pady=1)
        num_trans_button1.grid(row=2, column=1, sticky=W)
        num_trans_button2.grid(row=2, column=2, sticky=W)
        num_trans_button3.grid(row=2, column=3, sticky=W)
        num_scaling_label.grid(row=3, column=0, sticky=W, pady=1)
        num_scaling_button1.grid(row=3, column=1, sticky=W)
        num_scaling_button2.grid(row=3, column=2, sticky=W)
        num_scaling_button3.grid(row=3, column=3, sticky=W)
        num_new_checkbutton.grid(row=4, column=0, sticky=W, columnspan=2)
        num_apply_button.grid(row=5, column=3, sticky=W, pady=10)

        # Categorical column transformations
        self.__cat_col_frame = LabelFrame(self.__wrapper2, text='Categorical transformations:', font=self.__title_font)

        cat_column_label = Label(self.__cat_col_frame, text='Select column:', font=self.__label_font)
        self.__cat_column_combo = ttk.Combobox(self.__cat_col_frame, width=20, height=5)
        self.__cat_column_combo.state(["readonly"])
        self.fill_categorical_combobox()

        cat_null_label = Label(self.__cat_col_frame, text='1-Null values:', font=self.__label_font)
        self.__cat_null_control = IntVar()
        self.__cat_null_control.set(1)
        cat_null_button1 = ttk.Radiobutton(self.__cat_col_frame, value=1, variable=self.__cat_null_control,
                                           text="Drop row", command=self.default_null_button_handler)
        cat_null_button2 = ttk.Radiobutton(self.__cat_col_frame, value=2, variable=self.__cat_null_control,
                                           text="Default", command=self.default_null_button_handler)
        self.cat_null_default_entry = Entry(self.__cat_col_frame, width=9, state='disabled')

        cat_extract_label = Label(self.__cat_col_frame, text='2-Vectorizer (10):', font=self.__label_font)
        self.__cat_extract_control = IntVar()
        self.__cat_extract_control.set(3)
        cat_extract_button1 = ttk.Radiobutton(self.__cat_col_frame, value=1, variable=self.__cat_extract_control,
                                              text="Tfidf")
        cat_extract_button2 = ttk.Radiobutton(self.__cat_col_frame, value=2, variable=self.__cat_extract_control,
                                              text="Count")
        cat_extract_button3 = ttk.Radiobutton(self.__cat_col_frame, value=3, variable=self.__cat_extract_control,
                                              text="None")

        cat_apply_button = Button(self.__cat_col_frame, text='Apply', font=self.__button_font, width=7,
                                  command=self.transform_cat_col_click)

        # Packing elements into categorical column section
        cat_column_label.grid(row=0, column=0, sticky=W, pady=1)
        self.__cat_column_combo.grid(row=0, column=1, padx=5, sticky=W, columnspan=2)
        cat_null_label.grid(row=1, column=0, sticky=W, pady=1)
        cat_null_button1.grid(row=1, column=1, sticky=W)
        cat_null_button2.grid(row=1, column=2, sticky=W)
        self.cat_null_default_entry.grid(row=1, column=3, sticky=W)
        cat_extract_label.grid(row=2, column=0, sticky=W, pady=1)
        cat_extract_button1.grid(row=2, column=1, sticky=W)
        cat_extract_button2.grid(row=2, column=2, sticky=W)
        cat_extract_button3.grid(row=2, column=3, sticky=W)
        cat_apply_button.grid(row=3, column=3, sticky=W, pady=10)

        # Packing elements into wrapper2_frame
        self.__num_col_frame.grid(row=0, column=0, padx=5, sticky=N+S+E+W)
        self.__cat_col_frame.grid(row=0, column=1, padx=5, sticky=N+S+E+W)

        # Third section of the page
        self.__wrapper3 = LabelFrame(self.__step2_frame, text='Dataset Head (25 rows): ', font=self.__title_font)

        # Creating column summary table
        columns, col_width = list(self.__data_set.columns), []
        for col in columns:
            width = self.__data_set.head()[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 25)

        self.__data_table = ttk.Treeview(self.__wrapper3, columns=tuple(columns), show="headings", height=7)

        rows = self.__data_set.head(25).values
        for row in rows:
            self.__data_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__data_table.heading(i, text=col, anchor=W)
            self.__data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

        vsb2 = ttk.Scrollbar(self.__wrapper3, orient="vertical", command=self.__data_table.yview)
        hsb2 = ttk.Scrollbar(self.__wrapper3, orient="horizontal", command=self.__data_table.xview)
        self.__data_table.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)

        hsb2.pack(side=BOTTOM, fill='x', padx=5, pady=5)
        vsb2.pack(side=RIGHT, fill='y')
        self.__data_table.pack(side=TOP, fill='y', expand='yes', padx=5, anchor=W)

        # Navigation buttons
        self.__next_button = Button(self.__step2_frame, text='Next step', font=self.__button_font, width=14,
                                    command=self.unpack_frame_forward)
        self.__previous_button = Button(self.__step2_frame, text='Previous step', font=self.__button_font, width=14,
                                        command=self.unpack_frame_previous)

        # Packing all elements of the step2_frame
        project_title_label.pack(fill='x', pady=3)
        self.__wrapper1.pack(fill='x', expand='no', padx=10, pady=2)
        self.__wrapper2.pack(fill='both', expand='yes', padx=5, pady=2)
        self.__wrapper3.pack(fill='both', expand='yes', padx=10, pady=2)
        self.__previous_button.pack(side=LEFT, padx=10, pady=5)
        self.__next_button.pack(side=RIGHT, padx=10, pady=5)

    def generate_table_type(self):
        self.__column_types = self.__data_set.dtypes.reset_index().reset_index()
        self.__column_types.columns = ['#', 'Column', 'Type']
        self.__column_types['Null count'] = list(self.__data_set.isna().sum())
        self.__column_types['Non-Null count'] = self.__data_set.shape[0] - self.__column_types['Null count']
        self.__column_types = self.__column_types[['#', 'Column', 'Non-Null count', 'Null count', 'Type']]

    def refresh_type_table(self):
        self.generate_table_type()
        columns = list(self.__column_types.columns)

        col_widths = []
        for col in columns:
            width = self.__column_types[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_widths.append(width * 6 + 15)
            else:
                col_widths.append(len(str(col)) * 6 + 10)

        self.__columns_table.delete(*self.__columns_table.get_children())
        rows = self.__column_types.values
        for row in rows:
            self.__columns_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__columns_table.column(col, width=col_widths[i])

    def refresh_data_table(self):
        columns = list(self.__data_set.columns)
        col_width = []
        for col in columns:
            width = self.__data_set.head()[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 25)

        self.__data_table.delete(*self.__data_table.get_children())
        self.__data_table['columns'] = tuple(columns)

        rows = self.__data_set.head(25).values
        for row in rows:
            self.__data_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__data_table.heading(i, text=col, anchor=W)
            self.__data_table.column(i, width=col_width[i])

    def combo_selection(self, event):
        col_names = list(self.__data_set.columns)
        col_types = list(self.__data_set.dtypes)
        column = self.__column_combo.get()
        index = col_names.index(column)

        self.__name_control.set(column)
        self.__name_entry["textvariable"] = self.__name_control
        self.__type_combo.set(col_types[index])

    def metadata_click(self):
        col_names = list(self.__column_combo["values"])
        old_col_name = self.__column_combo.get()
        new_col_name = self.__name_entry.get()
        new_col_type = self.__type_combo.get()
        int_drop_col = self.__drop_option_var.get()

        col_names.remove(old_col_name)
        col_names.append(new_col_name)

        if len(self.__metadata_frame.grid_slaves()) > 8:
            self.__metadata_frame.grid_slaves()[0].destroy()

        if int_drop_col > 0:
            self.__data_set.drop(old_col_name, axis=1, inplace=True)
            col_names = list(self.__data_set.columns)
            types = list(self.__column_types['Type'])

            self.__column_combo["values"] = col_names
            self.__column_combo.set(col_names[0])

            self.__name_control.set(self.__column_combo.get())
            self.__name_entry["textvariable"] = self.__name_control

            self.__type_combo.set(types[0])
            self.__drop_option_var.set(0)

            self.fill_numeric_combobox()
            self.fill_categorical_combobox()
            self.refresh_type_table()
            self.refresh_data_table()

            # Check if its a vectorized column
            for dict_key in list(self.__vect_columns.keys()):
                if old_col_name in self.__vect_columns[dict_key]:
                    col_list = self.__vect_columns[dict_key]
                    col_list.remove(old_col_name)
                    self.__vect_columns[dict_key] = col_list

            success_label = Label(self.__metadata_frame, text='Column dropped successfully',
                                  font=self.__button_font)
            success_label.grid(row=4, column=0, columnspan=2, sticky=W)

        # Checking for duplicated column names
        elif len(col_names) == len(set(col_names)):
            try:
                self.__data_set[old_col_name] = self.__data_set[old_col_name].astype(new_col_type)
                self.__data_set.rename(columns={old_col_name: new_col_name}, inplace=True)
                col_names = list(self.__data_set.columns)
                index = col_names.index(new_col_name)
                self.__column_combo["values"] = col_names
                self.__column_combo.set(col_names[index])

                self.fill_categorical_combobox()
                self.fill_numeric_combobox()
                self.refresh_type_table()
                self.refresh_data_table()

                success_label = Label(self.__metadata_frame, text='Change applied successfully',
                                      font=self.__button_font)
                success_label.grid(row=4, column=0, columnspan=2, sticky=W)
            except Exception:
                error_label = Label(self.__metadata_frame, text='Error: invalid column type',
                                    font=self.__button_font, fg='red')
                error_label.grid(row=4, column=0, columnspan=2, sticky=W)
        else:
            error_label = Label(self.__metadata_frame, text='Error: repeated column name',
                                font=self.__button_font, fg='red')
            error_label.grid(row=4, column=0, columnspan=2, sticky=W)

    def fill_numeric_combobox(self):
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        columns = list(self.__data_set.select_dtypes(numeric_types).columns)
        columns.append('All columns')
        self.__num_column_combo["values"] = columns
        self.__num_column_combo.set(columns[0])

    def fill_categorical_combobox(self):
        categorical_types = ['object', 'category']
        columns = list(self.__data_set.select_dtypes(categorical_types).columns)
        columns.append('All columns')
        self.__cat_column_combo["values"] = columns
        self.__cat_column_combo.set(columns[0])

    def transform_num_col(self, column):
        if len(self.__num_col_frame.grid_slaves()) > 16:
            self.__num_col_frame.grid_slaves()[0].destroy()

        try:
            if self.__new_num_option_var.get() == 1:
                # Create new column to use base on current column
                self.__data_set[str(column + '2')] = self.__data_set[column]
                col_use = str(column + '2')
            else:
                col_use = str(column)

            # Checking how to handle Null values
            if self.__num_null_control.get() == 1:
                self.__data_set.dropna(subset=[col_use], inplace=True)
            elif self.__num_null_control.get() == 2:
                self.__data_set[col_use].fillna(value=self.__data_set[col_use].mean(), inplace=True)
            elif self.__num_null_control.get() == 3:
                self.__data_set[col_use].fillna(value=self.__data_set[col_use].median(), inplace=True)

            # Checking transformations required
            if self.__num_trans_control.get() == 1:
                self.__data_set[col_use] = self.__data_set[col_use].transform(np.sqrt)
            elif self.__num_trans_control.get() == 2:
                self.__data_set[col_use] = self.__data_set[col_use].transform(np.log)

            # Checking scaling required
            if self.__num_scaling_control.get() == 1:
                self.__data_set[col_use] = StandardScaler().fit_transform(np.array(self.__data_set[col_use]).reshape(-1, 1))
            elif self.__num_scaling_control.get() == 2:
                self.__data_set[col_use] = MinMaxScaler().fit_transform(np.array(self.__data_set[col_use]).reshape(-1, 1))

            success_label = Label(self.__num_col_frame, text='Transformation successful',
                                  font=self.__button_font)
            success_label.grid(row=5, column=0, columnspan=2, sticky=W)
        except Exception:
            error_label = Label(self.__num_col_frame, text='Error: invalid transformation',
                                font=self.__button_font, fg='red')
            error_label.grid(row=5, column=0, columnspan=2, sticky=W)

        # Refreshing view
        col_names = list(self.__data_set.columns)
        self.__column_combo["values"] = col_names
        self.fill_numeric_combobox()
        self.__num_column_combo.set(column)
        self.refresh_type_table()
        self.refresh_data_table()

    def transform_num_col_click(self):
        if self.__num_column_combo.get() == 'All columns':
            for num_col in self.__num_column_combo["values"][:-1]:
                self.transform_num_col(num_col)
            self.__num_column_combo.set('All columns')
        else:
            self.transform_num_col(self.__num_column_combo.get())

    def default_null_button_handler(self):
        if self.__cat_null_control.get() == 1:
            self.cat_null_default_entry.configure(state="disabled")
        elif self.__cat_null_control.get() == 2:
            self.cat_null_default_entry.configure(state="normal")

    def transform_cat_col_click(self):
        if self.__cat_column_combo.get() == 'All columns':
            for cat_col in self.__cat_column_combo["values"][:-1]:
                self.transform_cat_col(cat_col)
            self.__cat_column_combo.set('All columns')
        else:
            self.transform_cat_col(self.__cat_column_combo.get())

    def transform_cat_col(self, column):
        if len(self.__cat_col_frame.grid_slaves()) > 11:
            self.__cat_col_frame.grid_slaves()[0].destroy()

        try:
            col_use = str(column)
            # Checking how to handle Null values
            if self.__cat_null_control.get() == 1:
                self.__data_set.dropna(subset=[col_use], inplace=True)
            elif self.__cat_null_control.get() == 2:
                self.__data_set[col_use].fillna(value=str(self.cat_null_default_entry.get()), inplace=True)

            # Checking word vectorizer required
            dict_keys = list(self.__vect_columns.keys())
            if self.__cat_extract_control.get() == 1:
                if col_use in dict_keys:
                    print(self.__vect_columns[col_use])
                    self.__data_set.drop(self.__vect_columns[col_use], axis=1, inplace=True)

                count_vector = TfidfVectorizer(stop_words='english', lowercase=True, max_features=10)
                words_count = count_vector.fit_transform(self.__data_set[col_use].values)
                vectorized_columns = {col_use + '_' + col: words_count.A[:, i] for i, col in enumerate(count_vector.get_feature_names())}
                count_df = pd.DataFrame(vectorized_columns, index=self.__data_set.index)
                self.__data_set = pd.concat([self.__data_set, count_df], axis=1)

                self.__vect_columns[col_use] = list(vectorized_columns.keys())

            elif self.__cat_extract_control.get() == 2:
                if col_use in dict_keys:
                    self.__data_set.drop(self.__vect_columns[col_use], axis=1, inplace=True)

                count_vector = CountVectorizer(stop_words='english', lowercase=True, max_features=10)
                words_count = count_vector.fit_transform(self.__data_set[col_use].values)
                vectorized_columns = {col_use + '_' + col: words_count.A[:, i] for i, col in
                                      enumerate(count_vector.get_feature_names())}
                count_df = pd.DataFrame(vectorized_columns, index=self.__data_set.index)
                self.__data_set = pd.concat([self.__data_set, count_df], axis=1)

                self.__vect_columns[col_use] = list(vectorized_columns.keys())

            success_label = Label(self.__cat_col_frame, text='Transformation successful',
                                  font=self.__button_font)
            success_label.grid(row=3, column=0, columnspan=2, sticky=W)

        except Exception:
            error_label = Label(self.__cat_col_frame, text='Error: value must be a valid category',
                                font=self.__button_font, fg='red')
            error_label.grid(row=3, column=0, columnspan=2, sticky=W)

        # Refreshing view
        col_names = list(self.__data_set.columns)
        self.__column_combo["values"] = col_names
        self.fill_categorical_combobox()
        self.__cat_column_combo.set(column)
        self.refresh_type_table()
        self.refresh_data_table()

    def pack_frame(self):
        self.__step2_frame.pack(fill='both')

    def unpack_frame_previous(self):
        self.__step2_frame.pack_forget()
        self.__root.geometry('700x440')
        self.__previous_frame.pack(fill='both')

    def unpack_frame_forward(self):
        self.__step2_frame.pack_forget()
        Step3(self.__root, self.__data_set, self.__project_title, self.__step2_frame).pack_frame()
