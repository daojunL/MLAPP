from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from scipy.stats import probplot
import DataImporter


class ModelSummarizer:
    def __init__(self, root, x_test, y_test, model, project_title, previous_frame):
        self.__root = root
        self.__root.geometry('1000x1000')
        self.__label_font = tkfont.Font(family="Times New Roman", size=12)
        self.__button_font = tkfont.Font(family="Times New Roman", size=10)
        self.__title_font = tkfont.Font(family="Times New Roman", size=14)

        self.__x_test = x_test
        self.__y_test = y_test
        self.__model = model
        self.__project_title = project_title

        # Main frame of the Data modifier (Step 4)
        self.__previous_frame = previous_frame

        # Main frame of the Data explorer (Step 5)
        self.__step5_frame = Frame(self.__root)

        # Title of the page
        project_title_label = Label(self.__step5_frame, text="Project name: " + self.__project_title,
                                    font=tkfont.Font(family="Times New Roman", size=16))

        # First section of the page - Model Summary and Metrics
        self.__wrapper1 = LabelFrame(self.__step5_frame, text='Step 5: Model Summary', font=self.__title_font)

        self.__wrapper1.columnconfigure(0, weight=1)
        self.__wrapper1.columnconfigure(1, weight=1)

        self.__wrapper1_left = Frame(self.__wrapper1)
        self.__wrapper1_right = Frame(self.__wrapper1)

        # Summary frame (left) and residual table (right)
        self.__wrapper1_left.grid(row=0, column=0, sticky=N + S + W + E)
        self.__wrapper1_right.grid(row=0, column=1, sticky=N + S + W + E)

        # Constructing summary
        model_name = ""
        if type(self.__model).__name__ == "LinearRegression":
            model_name = "Linear Regression"
        elif type(self.__model).__name__ == "DecisionTreeRegressor":
            model_name = "Decision Tree Regession"
        elif type(self.__model).__name__ == "SVR":
            model_name = "Support Vector Regression"

        model_name_label = Label(self.__wrapper1_left, text="The model we use is " + model_name, font=self.__label_font)

        features_list = self.__x_test.columns.values.tolist()
        flag = False
        print(str(len(features_list)))
        if len(features_list) > 5:
            features_list = features_list[0:5]
            flag = True
        features_str = ", "
        features_str = features_str.join(features_list)
        if flag:
            features_label_text = "We have these features: " + features_str + "..."
        else:
            features_label_text = "We have these features: " + features_str
        features_label = Label(self.__wrapper1_left, text=features_label_text,
                               font=self.__label_font)

        y_label = Label(self.__wrapper1_left, text="The label (y) we use is " + self.__y_test.name,
                        font=self.__label_font)

        y_pred = self.__model.predict(self.__x_test)
        y_true = self.__y_test
        mse = round(mean_squared_error(y_true, y_pred, squared=False), 4)
        mse_label = Label(self.__wrapper1_left, text="The mean square error (MSE) is " + str(mse),
                          font=self.__label_font)
        r_square = round(r2_score(y_true, y_pred), 4)
        r2_label = Label(self.__wrapper1_left, text="The r square value (R^2) is " + str(r_square),
                         font=self.__label_font)
        exp_var_score = round(explained_variance_score(y_true, y_pred), 4)
        exp_var_score_label = Label(self.__wrapper1_left, text="The explained variance score is " + str(exp_var_score),
                                    font=self.__label_font)

        features_label.pack(fill="x")
        y_label.pack(fill="x")
        model_name_label.pack(fill="x")
        mse_label.pack(fill="x")
        r2_label.pack(fill="x")
        exp_var_score_label.pack(fill="x")

        # Constructing residual table
        table_label = Label(self.__wrapper1_right, text="Residual table:", font=self.__label_font)

        df = pd.DataFrame(
            {'Real Values': y_true, 'Predicted Values': y_pred, 'Residuals (Real - Predicted)': y_true - y_pred})

        columns, col_width = list(df.columns), []
        for col in columns:
            width = df.head()[col].astype('str').str.len().max()
            if width >= len(str(col)):
                col_width.append(width * 6 + 15)
            else:
                col_width.append(len(str(col)) * 6 + 25)

        self.__data_table = ttk.Treeview(self.__wrapper1_right, columns=tuple(columns), show="headings", height=5)

        rows = df.values
        for row in rows:
            self.__data_table.insert('', 'end', values=tuple(row))

        for i, col in enumerate(columns):
            self.__data_table.heading(i, text=col, anchor=W)
            self.__data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

        vsb = ttk.Scrollbar(self.__wrapper1_right, orient="vertical", command=self.__data_table.yview)
        self.__data_table.configure(yscrollcommand=vsb.set)

        table_label.pack(anchor=W)
        vsb.pack(side=RIGHT, fill='y')
        self.__data_table.pack(side=BOTTOM, fill='y', expand='yes', padx=5, anchor=W)

        # Second section of the page - Summary Plots
        self.__wrapper2 = LabelFrame(self.__step5_frame, text='Summary plots: ', font=self.__title_font)

        fig = plt.Figure(figsize=(10, 5), dpi=100)
        # First plot: Fitted vs Real Plot
        ax = fig.add_subplot(221)

        ax.scatter(x=y_true, y=y_pred, color=(0.2, 0.4, 0.6, 0.8), alpha=0.4)
        ax.set_xlabel("Real value", fontsize=10)
        ax.set_ylabel("Predicted value", fontsize=10)
        ax.set_title('Real vs. Predicted values', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)

        # Second plot: Q-Q residual plot
        ax = fig.add_subplot(222)

        residuals = y_true - y_pred
        std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        probplot(std_residuals, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor((0.2, 0.4, 0.6, 0.8))
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        ax.set_ylabel("Standardized Residuals", fontsize=10)
        ax.set_title('Normal Q-Q plot', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)

        # Third plot: Residuals vs Fitted Plot
        ax = fig.add_subplot(223)

        ax.scatter(x=y_pred, y=residuals, color=(0.2, 0.4, 0.6, 0.8), alpha=0.4)
        ax.set_xlabel("Predicted value", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.set_title('Predicted vs. Residuals', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)

        # Fourth plot: Scale - Location Plot
        ax = fig.add_subplot(224)

        sqrt_abs_std_residuals = np.sqrt(np.abs(std_residuals))
        ax.scatter(x=y_pred, y=sqrt_abs_std_residuals, color=(0.2, 0.4, 0.6, 0.8), alpha=0.4)
        ax.set_xlabel("Predicted value", fontsize=10)
        ax.set_ylabel("Sqrt(|Standardized Residuals|)", fontsize=10)
        ax.set_title('Scale - Location plot', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        fig.set_tight_layout(True)

        summary_plots = FigureCanvasTkAgg(fig, self.__wrapper2)
        summary_plots.get_tk_widget().pack()

        self.__next_button = Button(self.__step5_frame, text='Go back to step1', font=self.__button_font,
                                    width=14, command=self.unpack_frame_forward)
        self.__previous_button = Button(self.__step5_frame, text='Previous step', font=self.__button_font,
                                        width=14, command=self.unpack_frame_previous)

        project_title_label.pack(fill='x', pady=3)
        self.__wrapper1.pack(fill='x', expand='no', padx=10, pady=2)
        self.__wrapper2.pack(fill='x', expand='no', padx=10, pady=2)
        self.__previous_button.pack(side=LEFT, padx=10, pady=5)
        self.__next_button.pack(side=RIGHT, padx=10, pady=5)

    def pack_frame(self):
        self.__step5_frame.pack(fill='both')

    def unpack_frame_forward(self):
        self.__step5_frame.pack_forget()
        DataImporter.DataImporter(self.__root).pack_frame()

    def unpack_frame_previous(self):
        self.__step5_frame.pack_forget()
        self.__root.geometry('1000x550')  # MAYBE NEEDED TO BE CHANGED
        self.__previous_frame.pack(fill='both')