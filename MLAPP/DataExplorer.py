from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ModelTrainer import ModelTrainer


# Class in charge of exploring the data given the user inputs
class DataExplorer:
    def __init__(self, root, data_set, project_title, previous_frame):
        plt.style.use('seaborn')
        self.__root = root
        self.__root.geometry('1200x793')

        # Font objects to be used on the section
        self.__label_font = tkfont.Font(family="Times New Roman", size=12)
        self.__button_font = tkfont.Font(family="Times New Roman", size=10)
        self.__title_font = tkfont.Font(family="Times New Roman", size=14)

        # Data frame that with the loaded data
        self.__data_set = data_set
        self.__project_title = project_title

        # Main frame of the Data modifier (Step 2)
        self.__previous_frame = previous_frame

        # Main frame of the Data explorer (Step 3)
        self.__step3_frame = Frame(self.__root)

        # Title of the page
        project_title_label = Label(self.__step3_frame, text="Project name: " + self.__project_title,
                                    font=tkfont.Font(family="Times New Roman", size=16))

        # First section of the page - Columns descriptive statistics
        self.__wrapper1 = LabelFrame(self.__step3_frame, text='Step 3: Let\'s explore your data:',
                                     font=self.__title_font)
        self.__wrapper1.columnconfigure(0, weight=1, uniform="group")
        self.__wrapper1.columnconfigure(1, weight=1, uniform="group")

        # Numerical columns description section of the page
        self.__num_description_frame = Frame(self.__wrapper1)
        num_description_label = Label(self.__num_description_frame, text='Numerical columns:',
                                      font=self.__label_font)

        num_style = ttk.Style()
        num_style.configure("numstyle.Treeview.Heading", font=('Calibri', 11, 'bold'))
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        # Get descriptive statistic of numerical columns
        numeric_descriptions = self.__data_set.describe(include=numeric_types).round(2).reset_index()
        columns = list(numeric_descriptions.columns)
        col_width = []
        for col in columns:
            width = numeric_descriptions[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 20)

        self.__num_description_table = ttk.Treeview(self.__num_description_frame, columns=tuple(columns),
                                                    style="numstyle.Treeview", show="headings", height=8)
        rows = numeric_descriptions.values
        for row in rows:
            self.__num_description_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__num_description_table.heading(i, text=col, anchor=CENTER)
            self.__num_description_table.column(i, width=col_width[i], stretch=NO, anchor=E)

        # Changing first column
        self.__num_description_table.heading(0, text='', anchor=CENTER)
        self.__num_description_table.column(0, width=col_width[0], stretch=NO, anchor=W)

        # Creating scrollbar and set up
        hsb = ttk.Scrollbar(self.__num_description_frame, orient="horizontal",
                            command=self.__num_description_table.xview)
        self.__num_description_table.configure(xscrollcommand=hsb.set)

        num_description_label.pack(anchor=W)
        hsb.pack(side=BOTTOM, fill='x')
        self.__num_description_table.pack(side=TOP, expand='yes', anchor=W)

        # Categorical columns description section of the page
        self.__cat_description_frame = Frame(self.__wrapper1)
        cat_description_label = Label(self.__cat_description_frame, text='Categorical columns:',
                                      font=self.__label_font)
        cat_style = ttk.Style()
        cat_style.configure("catstyle.Treeview.Heading", font=('Calibri', 11, 'bold'))

        categorical_types = ['object', 'category']
        # Get descriptive statistic of categorical columns
        categorical_descriptions = self.__data_set.describe(include=categorical_types).reset_index()
        columns = list(categorical_descriptions.columns)
        col_width = []
        for col in columns:
            width = categorical_descriptions[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 20)

        self.__cat_description_table = ttk.Treeview(self.__cat_description_frame, columns=tuple(columns),
                                                    style="catstyle.Treeview", show="headings", height=4)
        rows = categorical_descriptions.values
        for row in rows:
            self.__cat_description_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__cat_description_table.heading(i, text=col, anchor=CENTER)
            self.__cat_description_table.column(i, width=col_width[i], stretch=NO, anchor=E)

        # Changing first column
        self.__cat_description_table.heading(0, text='', anchor=CENTER)
        self.__cat_description_table.column(0, width=col_width[0], stretch=NO, anchor=W)

        # Creating scrollbar and set up
        hsb2 = ttk.Scrollbar(self.__cat_description_frame, orient="horizontal",
                             command=self.__cat_description_table.xview)
        self.__cat_description_table.configure(xscrollcommand=hsb2.set)

        cat_description_label.pack(anchor=W)
        hsb2.pack(side=BOTTOM, fill='x')
        self.__cat_description_table.pack(side=TOP, expand='yes', anchor=N+W)

        # Packing both tables of the wrapper1
        self.__num_description_frame.grid(row=0, column=0, padx=10, sticky=N+W)
        self.__cat_description_frame.grid(row=0, column=1, padx=10, sticky=N+W)

        # Second section of the page - Exploratory plots
        self.__wrapper2 = Frame(self.__step3_frame)
        self.__wrapper2.columnconfigure(0, weight=1, uniform="group2")
        self.__wrapper2.columnconfigure(1, weight=1, uniform="group2")

        columns = list(self.__data_set.columns)

        # Single variable plots section
        self.__single_var_plot = LabelFrame(self.__wrapper2, text='Single variable exploration', font=self.__label_font)
        self.__single_var_setup = Frame(self.__single_var_plot)

        single_var_label = Label(self.__single_var_setup, text='Column:', font=self.__label_font)
        self.__single_var_combobox = ttk.Combobox(self.__single_var_setup, width=15)
        self.__single_var_combobox.state(["readonly"])
        self.__single_var_combobox["values"] = columns
        self.__single_var_combobox.set(columns[0])

        single_var_button = Button(self.__single_var_setup, text='Plot', font=self.__button_font, width=10,
                                   command=self.generate_single_var_plot)

        # Packing setup for single var plots
        single_var_label.pack(side=LEFT, padx=5, pady=5)
        self.__single_var_combobox.pack(side=LEFT, pady=5)
        single_var_button.pack(side=LEFT, padx=5, pady=5)

        self.__single_var_setup.pack(side=TOP, anchor=W)
        self.generate_single_var_plot()

        # Dual variable plots section
        self.__dual_var_plot = LabelFrame(self.__wrapper2, text='Dual variable exploration', font=self.__label_font)
        self.__dual_var_setup = Frame(self.__dual_var_plot)

        dual_var_label1 = Label(self.__dual_var_setup, text='Column 1:', font=self.__label_font)
        self.__dual_var_combobox1 = ttk.Combobox(self.__dual_var_setup, width=15)
        self.__dual_var_combobox1.state(["readonly"])
        self.__dual_var_combobox1["values"] = columns
        self.__dual_var_combobox1.set(columns[0])

        dual_var_label2 = Label(self.__dual_var_setup, text='Column 2:', font=self.__label_font)
        self.__dual_var_combobox2 = ttk.Combobox(self.__dual_var_setup, width=15)
        self.__dual_var_combobox2.state(["readonly"])
        self.__dual_var_combobox2["values"] = columns
        self.__dual_var_combobox2.set(columns[0])

        dual_var_button = Button(self.__dual_var_setup, text='Plot', font=self.__button_font, width=10,
                                 command=self.generate_dual_var_plot)

        # Packing setup for dual var plots
        dual_var_label1.pack(side=LEFT, padx=5, pady=5)
        self.__dual_var_combobox1.pack(side=LEFT, pady=5)
        dual_var_label2.pack(side=LEFT, padx=5, pady=5)
        self.__dual_var_combobox2.pack(side=LEFT, pady=5)
        dual_var_button.pack(side=LEFT, padx=5, pady=5)

        self.__dual_var_setup.pack(side=TOP, anchor=W)
        self.generate_dual_var_plot()

        # Packing both tables of the wrapper2
        self.__single_var_plot.grid(row=0, column=0, sticky=N+S+W+E)
        self.__dual_var_plot.grid(row=0, column=1, sticky=N+S+W+E, padx=3)

        # Navigation buttons
        self.__next_button = Button(self.__step3_frame, text='Next step', font=self.__button_font, width=14,
                                    command=self.unpack_frame_forward)
        self.__previous_button = Button(self.__step3_frame, text='Previous step', font=self.__button_font, width=14,
                                        command=self.unpack_frame_previous)

        # Packing all elements of the step3_frame
        project_title_label.pack(fill='x', pady=3)
        self.__wrapper1.pack(fill='x', expand='no', padx=10, pady=2)
        self.__wrapper2.pack(fill='x', expand='no', padx=10, pady=2)
        self.__previous_button.pack(side=LEFT, padx=10, pady=5)
        self.__next_button.pack(side=RIGHT, padx=10, pady=5)

    def generate_single_var_plot(self):
        # Destroy last single variable generated plot
        if len(self.__single_var_plot.pack_slaves()) > 1:
            self.__single_var_plot.pack_slaves()[-1].destroy()

        col_name = str(self.__single_var_combobox.get())
        if self.__data_set[col_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            # Generate histogram of numerical column
            self.plot_histogram(str(col_name))

        elif self.__data_set[col_name].dtype in ['object', 'category']:
            # Generate count plot of categorical column
            self.plot_count_bar(str(col_name))

    def plot_histogram(self, col):
        fig = plt.Figure(figsize=(6.5, 3.9), dpi=100)
        ax = fig.add_subplot(111)

        ax.hist(self.__data_set[col].values, density=True, bins=20, color=(0.2, 0.4, 0.6, 0.8), alpha=0.75)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title('Histogram of ' + col, fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)

        histogram = FigureCanvasTkAgg(fig, self.__single_var_plot)
        histogram.get_tk_widget().pack(side=LEFT, fill=BOTH, padx=5, pady=5)

    def plot_count_bar(self, col):
        fig = plt.Figure(figsize=(6.5, 3.9), dpi=100)
        ax = fig.add_subplot(111)

        cat_counts = self.__data_set[col].value_counts(normalize=True, sort=True, ascending=False)
        ax.bar(cat_counts.index, cat_counts.values, color=(0.2, 0.4, 0.6, 0.8), alpha=0.75)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title('Count plot of ' + col, fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        if len(cat_counts.index) > 5:
            ax.tick_params(axis='x', rotation=90, labelsize=7)
        fig.set_tight_layout(True)
        count_plot = FigureCanvasTkAgg(fig, self.__single_var_plot)
        count_plot.get_tk_widget().pack(side=LEFT, fill=BOTH, padx=5, pady=5)

    def generate_dual_var_plot(self):
        # Destroy last dual variable generated plot
        if len(self.__dual_var_plot.pack_slaves()) > 1:
            self.__dual_var_plot.pack_slaves()[-1].destroy()

        col1_name = str(self.__dual_var_combobox1.get())
        col2_name = str(self.__dual_var_combobox2.get())

        # Check type of selected columns
        if self.__data_set[col1_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            col1_type = 'num'
        else:
            col1_type = 'cat'

        if self.__data_set[col2_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            col2_type = 'num'
        else:
            col2_type = 'cat'

        if col1_type == 'num' and col2_type == 'num':
            # Generate scatterplot of two numerical columns
            self.plot_scatter_plot(col1_name, col2_name)
        elif col1_type == 'num' and col2_type == 'cat':
            # Generate boxplot of one categorical and numerical column
            self.plot_boxplot(col1_name, col2_name)
        elif col1_type == 'cat' and col2_type == 'num':
            # Generate boxplot of one categorical and numerical column
            self.plot_boxplot(col2_name, col1_name)
        elif col1_type == 'cat' and col2_type == 'cat':
            # Generate stacked bar plot of two categorical columns
            self.plot_stacked_bar(col1_name, col2_name)

    def plot_scatter_plot(self, col1, col2):
        fig = plt.Figure(figsize=(6.5, 3.9), dpi=100)
        ax = fig.add_subplot(111)

        ax.scatter(x=self.__data_set[col1].values, y=self.__data_set[col2].values,
                   color=(0.2, 0.4, 0.6, 0.8), alpha=0.4)
        ax.set_xlabel(col1, fontsize=10)
        ax.set_ylabel(col2, fontsize=10)
        ax.set_title('Scatter plot of ' + col1 + ' vs. ' + col2, fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)

        scatter_plot = FigureCanvasTkAgg(fig, self.__dual_var_plot)
        scatter_plot.get_tk_widget().pack(side=LEFT, fill=BOTH, padx=5, pady=5)

    def plot_boxplot(self, col1, col2):
        fig = plt.Figure(figsize=(6.5, 3.9), dpi=100)
        ax = fig.add_subplot(111)

        self.__data_set.boxplot(column=col1, by=col2, ax=ax)
        ax.set_xlabel(col2, fontsize=10)
        ax.set_ylabel(col1, fontsize=10)
        ax.set_title('Boxplot of ' + col1 + ' by ' + col2, fontsize=11)
        fig.suptitle('')
        ax.tick_params(axis='both', labelsize=8)
        if self.__data_set[col2].nunique() > 5:
            ax.tick_params(axis='x', rotation=90, labelsize=7)
        fig.set_tight_layout(True)

        boxplot = FigureCanvasTkAgg(fig, self.__dual_var_plot)
        boxplot.get_tk_widget().pack(side=LEFT, fill=BOTH, padx=5, pady=5)

    def plot_stacked_bar(self, col1, col2):
        fig = plt.Figure(figsize=(6.5, 3.9), dpi=100)
        ax = fig.add_subplot(111)

        cross_table = pd.crosstab(self.__data_set[col1], self.__data_set[col2], normalize='index')
        cross_table.plot.bar(stacked=True, ax=ax)

        ax.set_xlabel(col1, fontsize=10)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title('Stacked bars plot of ' + col1 + ' by ' + col2, fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        if self.__data_set[col1].nunique() > 5:
            ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6)
        fig.set_tight_layout(True)

        stacked_bars = FigureCanvasTkAgg(fig, self.__dual_var_plot)
        stacked_bars.get_tk_widget().pack(side=LEFT, fill=BOTH, padx=5, pady=5)

    def pack_frame(self):
        # Pack the data explorer main frame
        self.__step3_frame.pack(fill='both')

    def unpack_frame_forward(self):
        # Unpack the data explorer main frame
        self.__step3_frame.pack_forget()
        ModelTrainer(self.__root, self.__data_set, self.__project_title, self.__step3_frame).pack_frame()

    def unpack_frame_previous(self):
        # Unpack the data explorer main frame
        self.__step3_frame.pack_forget()
        self.__root.geometry('790x730')

        # Call the previous step main frame (Data modifier)
        self.__previous_frame.pack(fill='both')
