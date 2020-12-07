from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class ModelSummary:
    def __init__(self, root, x_test, y_test, model, project_title, previous_frame):
        self.__root = root
        self.__root.geometry('1000x550')
        self.__label_font = tkfont.Font(family="Times New Roman", size=12)
        self.__button_font = tkfont.Font(family="Times New Roman", size=10)
        self.__title_font = tkfont.Font(family="Times New Roman", size=14)

        self.__x_test = x_test
        self.__y_test = y_test
        self.__model = model
        self.__project_title = project_title
        self.__previous_frame = previous_frame

        self.__ModelSummary_frame = Frame(self.__root)

        # First section of the page
        self.__wrapper1 = LabelFrame(self.__ModelSummary_frame, text='Step 5: Model Summary', font=self.__title_font)
        project_name_label = Label(self.__wrapper1, text="The project name is " + self.__project_title, font=self.__label_font)
        model_name  = ""
        if type(self.__model).__name__ == "LinearRegression":
            model_name = "Linear Regression"
        elif type(self.__model).__name__ == "DecisionTreeRegressor":
            model_name = "Decision Tree Regession"
        elif type(self.__model).__name__ == "SVR":
            model_name = "SVM"

        model_name_label = Label(self.__wrapper1, text="The model name is " + model_name, font=self.__label_font)

        features_list = self.__x_test.columns.values.tolist()
        features_str = ", "
        features_str = features_str.join(features_list)
        features_label = Label(self.__wrapper1, text="We have these features:  " + features_str, font=self.__label_font)

        y_label = Label(self.__wrapper1, text="The label (y) we use is " + self.__y_test.name, font=self.__label_font)

        y_pred = self.__model.predict(self.__x_test)
        y_true = self.__y_test
        mse = mean_squared_error(y_true, y_pred, squared=False)
        mse_label = Label(self.__wrapper1, text="The mean square error (MSE) is " + str(mse), font=self.__label_font)

        project_name_label.pack(fill="x")
        model_name_label.pack(fill="x")
        features_label.pack(fill="x")
        y_label.pack(fill="x")
        mse_label.pack(fill="x")

        # second section
        self.__wrapper2 = LabelFrame(self.__ModelSummary_frame, text='Table of real value of y and predicted value of y: ', font=self.__title_font)
        df = pd.DataFrame({'Real Values': self.__y_test, 'Predicted Values': y_pred})
        columns, col_width = list(df.columns), []
        for col in columns:
            width = df.head()[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 25)

        self.__data_table = ttk.Treeview(self.__wrapper2, columns=tuple(columns), show="headings", height=7)

        rows = df.values
        for row in rows:
            self.__data_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__data_table.heading(i, text=col, anchor=W)
            self.__data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

        hsb2 = ttk.Scrollbar(self.__wrapper2, orient="horizontal", command=self.__data_table.xview)
        self.__data_table.configure(xscrollcommand=hsb2.set)

        hsb2.pack(side=BOTTOM, fill='x', padx=5, pady=5)
        self.__data_table.pack(side=TOP, fill='y', expand='yes', padx=5, anchor=W)

        # third section - 2 * 2 grid
        self.__wrapper3 = Frame(self.__ModelSummary_frame)
        self.__wrapper3.columnconfigure(0, weight=1)
        self.__wrapper3.columnconfigure(1, weight=1)
        self.__wrapper3.rowconfigure(0, weight=1)
        self.__wrapper3.rowconfigure(1, weight=1)
        # place the first scatterplot
        self.__pred_y_scaplot = LabelFrame(self.__wrapper3, text='Plot 1', font=self.__label_font)
        self.__pred_frame = Frame(self.__pred_y_scaplot)
        fig = plt.Figure(figsize=(6.5, 3.9), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x=y_true, y=y_pred, color=(0.2, 0.4, 0.6, 0.8), alpha=0.4)
        ax.set_xlabel("Real y", fontsize=10)
        ax.set_ylabel("Predicted y", fontsize=10)
        ax.set_title('Scatter plot of real y and predicted y', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)
        scatter_plot = FigureCanvasTkAgg(fig, self.__pred_frame)  # is the frame
        scatter_plot.get_tk_widget().pack(side=LEFT, fill=BOTH, padx=5, pady=5)
        self.__pred_frame.pack(side=TOP, anchor=W)

        # Packing four tables of the wrapper3
        self.__pred_y_scaplot.grid(row=0, column=0, sticky=N+S+W+E)

        ##### --- TO DO: Add other plots ---- ###
        #########################################

        self.__next_button = Button(self.__ModelSummary_frame, text='Go back to step1', font=self.__button_font, width=14, command=self.unpack_frame_forward)
        self.__previous_button = Button(self.__ModelSummary_frame, text='Previous step', font=self.__button_font, width=14, command=self.unpack_frame_previous)

        self.__wrapper1.pack(fill='x', expand='no', padx=10, pady=2)
        self.__wrapper2.pack(fill='x', expand='no', padx=10, pady=2)
        self.__wrapper3.pack(fill='x', expand='no', padx=10, pady=2)
        self.__previous_button.pack(side=LEFT, padx=10, pady=5)
        self.__next_button.pack(side=RIGHT, padx=10, pady=5)


    def pack_frame(self):
        self.__ModelSummary_frame.pack(fill='both')

    def unpack_frame_forward(self):
        # go back to the step 1
        pass
        # self.__step4_frame.pack_forget()
        # ModelSummary(self.__root, self.__data_set, self.__project_title, self.__step4_frame).pack_frame()



    def unpack_frame_previous(self):
        self.__ModelSummary_frame.pack_forget()
        self.__root.geometry('1000x550') # MAYBE NEEDED TO BE CHANGED
        self.__previous_frame.pack(fill='both')