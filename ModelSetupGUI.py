from tkinter import *
from tkinter import ttk
import tkinter.font as tkfont
import pandas as pd

def main():
    root = Tk()
    root.title("ML APP")
    root.resizable(width=False, height=True)
    root.geometry('700x275')

    #### pass some parameter from previous function ####
    data = pd.read_table("/Users/daojun/PycharmProjects/CS5007/venv/MLAPP/Recipes.csv", sep=";", header=1)
    data = data.drop_duplicates().head() # get the first 5 lines to display


    # TO DO - Code Style: - we can use another file to store this, and use it in our entire project  (macro)
    #########################################################################################################
    # languages
    summary_title = "Let us look at the current data"
    create_model = "Step4: Model Setup"
    choose_ratio = "choose the train/test ratio"
    choose_model = "choose the model"
    choose_metrics = "choose evaluation metrics"
    project_title = "project title"

    ##########################################################################################################
    # font
    app_title_font = tkfont.Font(family="Times New Roman", size=16)
    label_font = tkfont.Font(family="Times New Roman", size=12)
    frame_title_font = tkfont.Font(family="Times New Roman", size=14)
    #########################################################################################################

    # Title of the page
    project_title_label = Label(root, text="Project name: " + project_title,
                                font = app_title_font)
    # display the current data
    display_frame = Frame(root)
    table_frame = LabelFrame(display_frame, text=summary_title,font=app_title_font)

    data_table = ttk.Treeview(table_frame, columns=tuple(columns), show="headings", height=5)

    rows = data.head().values
    for row in rows:
        data_table.insert('', 'end', values=tuple(row))

    for i, col in enumerate(columns):
        data_table.heading(i, text=col, anchor=W)
        data_table.column(col, width=col_width[i], stretch=NO, anchor=W)

    hsb = ttk.Scrollbar(self.__wrapper3, orient="horizontal", command=data_table.xview)
    data_table.configure(xscrollcommand=hsb.set)
    data_table = ttk.Treeview(self.__wrapper3, columns=tuple(columns), show="headings", height=5)



    # variables
    ratio = [1, 2, 3, 4]
    model = ["Regression", "Knn"]
    eva_metrics = ["MAE","MSE","RMSE"]

    # create frame
    model_gui_frame = Frame(root)

    # create first section - input
    input_frame = LabelFrame(model_gui_frame, text= create_model, font=frame_title_font)

    # choose the train/test ratio label & combo box
    label_ratio = Label(input_frame, text=choose_ratio, font=label_font)
    combo_box_ratio = ttk.Combobox(input_frame, width=20, height=5)
    combo_box_ratio.state(["readonly"])
    combo_box_ratio["values"] = ratio
    combo_box_ratio.set(ratio[0])

    # choose the model label & combo box
    label_model = Label(input_frame, text=choose_model, font=label_font)
    combo_box_model = ttk.Combobox(input_frame, width=20, height=5)
    combo_box_model.state(["readonly"])
    combo_box_model["values"] = model
    combo_box_model.set(model[0])

    # choose the model label & combo box
    label_eva_metrics = Label(input_frame, text=choose_metrics, font=label_font)
    combo_box_eva_metrics = ttk.Combobox(input_frame, width=20, height=5)
    combo_box_eva_metrics.state(["readonly"])
    combo_box_eva_metrics["values"] = eva_metrics
    combo_box_eva_metrics.set(eva_metrics[0])

    # Packing of elements into the section
    label_ratio.pack(side=LEFT, padx=3, pady=5)
    combo_box_ratio.pack(side=LEFT, padx=3)
    label_model.pack(side=LEFT, padx=3)
    combo_box_model.pack(side=LEFT)

    # display the label using pack layout
    input_frame.pack(fill='x', expand='no', padx=20, pady=5)

    model_gui_frame.pack(fill='both')


    root.mainloop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()