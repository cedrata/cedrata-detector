# ###########################################################################
# Author: Luca Greggio
# Github: ...
# Email: luca01.greggio@edu.unife.it, greggius9891@gmail.com
# This is a machine learning project able to detect where notes are played 
# in audio tracks. 
# If you are interested in it please please beofre using this code contact me
# ###########################################################################

# This is the file that gives functionality to the ui

# python modules
import os
import sys
import pandas as pd
from numpy import source
from functools import partial

# PyQt related modules
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator, QValidator
from PyQt5.QtCore import QThread, pyqtSignal, qCritical
from PyQt5.QtWidgets import QLineEdit, QMainWindow, QFileDialog, QMessageBox

# Weka related modules
import weka.core.jvm as jvm

# Local modules
from ui_design import Ui_MainWindow
from extraction import Classification, Prediction
from settings import checkbox as default_checkbox, window_len as default_window_len, overlap as default_overlap, samplerate as default_samplerate

# Main window UI Setup and logics
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the user interface from Designer
        self.setupUi(self)

        # The progress bar will be shown only after the operation are completed
        # immediatly after hidden again

        # The progress bar has a maximum value
        # it refers to the operatoins needed to complete the job after hitting the "start" button
        self.classify_start_progressBar.hide()
        self.classify_start_progressBar.setMaximum(6)
        self.predict_start_progressBar.hide()
        self.predict_start_progressBar.setMaximum(7)
        self.merge_start_progressBar.hide()
        # Maximum for merge progress bar is variable
        # will be setted inside def start_merge
        # self.merge_start_progressBar.setMaximum()
        
        # Initializing settings

        # Validating settings input
        uint_validator = QRegExpValidator(QRegExp(r'[0-9]+'))
        self.settings_classify_window_len_lineEdit.setValidator(uint_validator)
        self.settings_classify_overlap_lineEdit.setValidator(uint_validator)
        # self.settings_classify_samplerate_lineEdit.setValidator(uint_validator)
        self.settings_predict_window_len_lineEdit.setValidator(uint_validator)
        self.settings_predict_overlap_lineEdit.setValidator(uint_validator)
        # self.settings_predict_samplerate_lineEdit.setValidator(uint_validator)

        # Classify section
        # Checkboxes
        # Mean
        self.settings_classify_mean_checkbox.setChecked(default_checkbox()) # True
        # Variance
        self.settings_classify_variance_checkbox.setChecked(default_checkbox()) # True
        # Mean der 1
        self.settings_classify_1_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Mean der 2
        self.settings_classify_2_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Window lenght
        self.settings_classify_window_len_lineEdit.setText(str(default_window_len()))
        # Window overlap
        self.settings_classify_overlap_lineEdit.setText(str(default_overlap()))
        # Samplerate
        # self.settings_classify_samplerate_lineEdit.setText(str(default_samplerate()))
        # Predict section
        # Mean
        self.settings_predict_mean_checkbox.setChecked(default_checkbox()) # True
        # Variance
        self.settings_predict_variance_checkbox.setChecked(default_checkbox()) # True
        # Mean der 1
        self.settings_predict_1_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Mean der 2
        self.settings_predict_2_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Window lenght
        self.settings_predict_window_len_lineEdit.setText(str(default_window_len()))
        # Window overlap
        self.settings_predict_overlap_lineEdit.setText(str(default_overlap()))
        # Samplerate
        # self.settings_predict_samplerate_lineEdit.setText(str(default_samplerate()))
        
        # Disabling not needed seettings buttons
        # When a value will be modified, the buttons will be enabled
        self.settings_apply_button.setDisabled(True)
        self.settings_reset_button.setDisabled(True)

        # Events bindings
        
        # Partial function creates a call to a desired function with the needed arguments
        # the partial is then called inside the ...connect(partial)...
        # now is possibles to call the needed functions inside other functions with arguemtns.
        
        # Classify tab bindings
        # Select source folder containing audio to classify
        csb = partial(self.select_folder, "classify_source_browse")
        self.classify_source_browse.clicked.connect(csb)
        # Select destination arff file to store classification
        cdb = partial(self.select_folder, "classify_destination_browse")
        self.classify_destination_browse.clicked.connect(cdb)
        # Start classification
        self.classify_start_button.clicked.connect(self.start_classification)
        
        # Predict tab bindings
        # Select audio file where to predict notes
        self.predict_source_browse.clicked.connect(self.select_audio_file)
        # Select the *.arff file to use as a classifier to predict notes in the selected file
        self.predict_relation_browse.clicked.connect(self.select_arff_file)
        # Select folder where to store predicted data
        pbd = partial(self.select_folder, "predict_destination_browse")
        self.predict_destination_browse.clicked.connect(pbd)
        # Start prediction
        self.predict_start_button.clicked.connect(self.start_prediction)

        # Merge tab bindings
        self.merge_select_button.clicked.connect(self.select_csv_files)
        self.merge_destination_browse.clicked.connect(self.save_csv_files)
        self.merge_start_button.clicked.connect(self.start_merge)

        # Settings tab bindings
        # Applying settings
        self.settings_apply_button.clicked.connect(self.apply_settings)
        # Reset settings
        self.settings_reset_button.clicked.connect(self.reset_settings)
        # whenever any setting is modified enable the apply and reset buttons
        self.settings_classify_mean_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_classify_variance_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_classify_1_deg_der_mean_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_classify_2_deg_der_mean_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_classify_window_len_lineEdit.textChanged.connect(self.settings_buttons_toggle)
        self.settings_classify_overlap_lineEdit.textChanged.connect(self.settings_buttons_toggle)
        # self.settings_classify_samplerate_lineEdit.textChanged.connect(self.settings_buttons_toggle)
        self.settings_predict_mean_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_predict_variance_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_predict_1_deg_der_mean_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_predict_2_deg_der_mean_checkbox.clicked.connect(self.settings_buttons_toggle)
        self.settings_predict_window_len_lineEdit.textChanged.connect(self.settings_buttons_toggle)
        self.settings_predict_overlap_lineEdit.textChanged.connect(self.settings_buttons_toggle)
        # self.settings_predict_samplerate_lineEdit.textChanged.connect(self.settings_buttons_toggle)
    # def __init__

    # Event Bindings
    # To prevent default behaviour on the exit button (right up corner)
    # this functions has been create. It stops the jvm before quitting.
    def closeEvent(self, event):
        print("User has clicked the red x on the main window")
        jvm.stop()
        event.accept()
    # def closeEvent

    # Display a message into a QMessageBox, it can be Error, Warning or Success.
    # It contains informations and a detailed description
    # Info is a non-detailed message, informations
    # Descriprion is a detailed description and informations about the message to display
    # msg_type is to decide weather is a success or error, must be a string
    def display_message(self, info: str, description: str, msg_type: str):
        msg = QMessageBox()
        msg.setMaximumWidth(400)
        msg.setMaximumHeight(200)
        msg.setIcon(QMessageBox.Critical) if msg_type == "Error" else\
            (msg.setIcon(QMessageBox.Warning) if msg_type == "Warning" else\
                msg.setIcon(QMessageBox.Information)) # else "Success/Information"
        msg.setText(info)
        msg.setInformativeText(description)
        msg.setWindowTitle("Error") if msg_type == "Error" else\
            (msg.setWindowTitle("Warning") if msg_type == "Warning" else\
                msg.setWindowTitle("Success")) # else "Success/Information"
        msg.exec_()
    # def display_message

    # Selecting folder where to store or retrive data.
    # This function is used in classify and in predict tabs, 
    # is important to know which button has been clicked to know where
    # to retrive the text from
    def select_folder(self, btn_name: str):
        try:
            origin = os.getenv("HOME")
            title = "Chose Directory"
            dir_path = QFileDialog.getExistingDirectory(self, title, directory=origin)
            if btn_name == "classify_source_browse":
                self.classify_source_path.setText(dir_path)
            elif btn_name == "classify_destination_browse":
                self.classify_destination_path.setText(dir_path)
            elif btn_name == "predict_destination_browse":
                self.predict_destination_path.setText(dir_path)
        except Exception as e:
            info = "General error"
            description = str(e)
            self.display_message(info, description, "Error")
        # except
    # def select_folder

    # Selecting auido file where to apply prediction.
    def select_audio_file(self):
        try:
            origin = os.getenv("HOME")
            title = "Select Audio File"
            filters = "All Uncompressed Audio(*.aif | *.aiff | *.wav);;aiff Files(*.aiff);;aif Files(*.aif);;wav Files(*.wav)"
            file_path, _ = QFileDialog.getOpenFileName(self, title, origin, filters)
            self.predict_source_path.setText(file_path)
        except Exception as e:
            info = "General error"
            description = str(e)
            self.display_message(info, description, "Error")
        # except
    # def select_audio_file

    # Selecting arff file where to store or get data to do predictions.
    def select_arff_file(self):
        try:
            origin = os.getenv("HOME")
            title = "Select Arff File"
            filters = "arff Files(*.arff)"
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.AnyFile)
            file_path, _ = dialog.getOpenFileName(self, title, origin, filters)
            self.predict_relation_path.setText(file_path)
        except Exception as e:
            info = "General error"
            description = str(e)
            self.display_message(info, description, "Error")
        # except
    # def select_arff_file

    # Select multiple csv files, used to merge csv files together
    def select_csv_files(self):
        try:
            origin = os.getenv("HOME")
            title = "Select CSV files"
            filters = "csv Files(*.csv)"
            dialog = QFileDialog()
            files_paths, _ = dialog.getOpenFileNames(self, title, origin, filters)
            for file_path in files_paths:
                # print(file_path, type(file_path))
                if file_path not in self.merge_paths_list.toPlainText():
                    self.merge_paths_list.append(file_path)
                #if
            # for
        except Exception as e:
            info = "General error"
            description = str(e)
            self.display_message(info, description, "Error")
        # except
    # def select_csv_files

    # Select destination where save csv file, in predict
    def save_csv_files(self):
        try:
            origin = os.getenv("HOME")+"/untitled.csv"
            title = "Select CSV files"
            filters = "csv Files(*.csv)"
            dialog = QFileDialog()
            file_path, _ = dialog.getSaveFileName(self, title, origin, filters)
            self.merge_destination_path.setText(file_path)
        except Exception as e:
            info = "General error"
            description = str(e)
            self.display_message(info, description, "Error")
        # except

    # Start classification of audio files contained into a folder.
    # All wav audio files contained in a folder will be imported
    # to extract their features, then those will be stored 
    # in the arff selected file.
    def start_classification(self):
        if self.settings_apply_button.isEnabled():
            info = "Settings Error"
            description = "Please apply the settings before starting the classification"
            self.display_message(info, description, "Error")
            # exit
            return
        # if
        # Initlize and show progress bar
        prog_count = 0

        # Progress bar
        self.classify_start_button.hide()
        self.classify_start_progressBar.show()
        self.classify_start_progressBar.setValue(prog_count)

        # Get fileds text
        csp = self.classify_source_path.text().strip() # Folder
        crn = self.classify_relation_name.text().strip().lower()
        cdp = self.classify_destination_path.text().strip() # File
        prog_count += 1
        self.classify_start_progressBar.setValue(prog_count)

        # Get settings
        try:
            smean = self.settings_classify_mean_checkbox.isChecked()
            svar = self.settings_classify_variance_checkbox.isChecked()
            s1der = self.settings_classify_1_deg_der_mean_checkbox.isChecked()
            s2der = self.settings_classify_2_deg_der_mean_checkbox.isChecked()
            win_len = int(self.settings_classify_window_len_lineEdit.text())
            overlap = int(self.settings_classify_overlap_lineEdit.text())
        except Exception as e:
            info = "Settings Error"
            description = str(e)
            self.display_message(info, description, "Error")
            self.classify_start_progressBar.hide()
            self.classify_start_button.show()
            # exit
            return
        # except

        # A generale Exception is used, every exceptions leads to the
        # same conclusion, display a custom error message in a QMessageBox
        try:
            # Extracting features
            extraction = Classification()

            extraction.find_audio(csp)
            prog_count += 1
            self.classify_start_progressBar.setValue(prog_count)

            for wav_file in extraction.wav_files:
                extraction.calc_features(wav_file, smean, svar, s1der, s2der, win_len, overlap)
            # for
            prog_count += 1
            self.classify_start_progressBar.setValue(prog_count)

            extraction.norm_features(smean, svar, s1der, s2der)
            prog_count += 1
            self.classify_start_progressBar.setValue(prog_count)

            extraction.save_arff(crn, cdp, smean, svar, s1der, s2der)
            prog_count += 1
            self.classify_start_progressBar.setValue(prog_count)
        except Exception as e:
            info = "Classification Error"
            description = str(e)
            self.display_message(info, description, "Error")
            self.classify_start_progressBar.hide()
            self.classify_start_button.show()
            # exit
            return
        # exceptions
        
        # Clearing input
        self.classify_source_path.setText("")
        self.classify_relation_name.setText("")
        self.classify_destination_path.setText("")
        prog_count += 1
        self.classify_start_progressBar.setValue(prog_count)

        info = "Operation Completed"
        description = "Files have been created succesfully in " + cdp
        self.display_message(info, description, "Success")

        self.classify_start_progressBar.hide()
        self.classify_start_button.show()
    # def start_classification


    # Start the prediction of a selected audio file
    def start_prediction(self):
        if self.settings_apply_button.isEnabled():
            info = "Settings Error"
            description = "Please apply the settings before starting the prediction"
            self.display_message(info, description, "Error")
            # exit
            return
        #if

        # Initialize progress bar
        prog_count = 0
        
        # Progress bar
        self.predict_start_button.hide()
        self.predict_start_progressBar.show()
        self.predict_start_progressBar.setValue(prog_count)

        # Get fields texts
        psp = self.predict_source_path.text().strip() # File
        prp = self.predict_relation_path.text().strip() # File
        pdp = self.predict_destination_path.text().strip() # Folder
        prog_count += 1
        self.predict_start_progressBar.setValue(prog_count)

        try:
            # Get settings
            smean = self.settings_predict_mean_checkbox.isChecked()
            svar = self.settings_predict_variance_checkbox.isChecked()
            s1der = self.settings_predict_1_deg_der_mean_checkbox.isChecked()
            s2der = self.settings_predict_2_deg_der_mean_checkbox.isChecked()
            win_len = int(self.settings_predict_window_len_lineEdit.text())
            overlap = int(self.settings_predict_overlap_lineEdit.text())
        except Exception as e:
            info = "Settings Error"
            description = str(e)
            self.display_message(info, description, "Error")
            self.predict_start_progressBar.hide()
            self.predict_start_button.show()
            # exit
            return
        # except

        try:
            # Predicting
            p = Prediction()

            p.calc_features(psp, smean, svar, s1der, s2der, win_len, overlap, "?")
            prog_count += 1
            self.predict_start_progressBar.setValue(prog_count)

            p.norm_features(smean, svar, s1der, s2der)
            prog_count += 1
            self.predict_start_progressBar.setValue(prog_count)

            # getting the name of the *.arff file, to name the result after it
            rname = prp.split("/")
            rname = rname[len(rname)-1]
            rname = rname.split(".")
            rname = rname[0]
            tmp = p.save_arff(rname, "/tmp/", smean, svar, s1der, s2der)
            prog_count += 1
            self.predict_start_progressBar.setValue(prog_count)

            p.predict(tmp, prp)
            prog_count += 1
            self.predict_start_progressBar.setValue(prog_count)

            p.save_prediction(rname,pdp)
            prog_count += 1
            self.predict_start_progressBar.setValue(prog_count)
        except Exception as e:
            info = "Prediction Error"
            description = str(e)
            self.display_message(info, description, "Error")
            self.predict_start_progressBar.hide()
            self.predict_start_button.show()
            # exit
            return
        # except

        # Clearing input
        self.predict_source_path.setText("")
        self.predict_relation_path.setText("")
        self.predict_destination_path.setText("")

        prog_count += 1
        self.predict_start_progressBar.setValue(prog_count)

        info = "Operation Completed"
        description = "Files have been created succesfully in " + pdp
        self.display_message(info, description, "Success")

        self.predict_start_progressBar.hide()
        self.predict_start_button.show()
    # def start_prediction

    # Start the merging operation on csv files
    def start_merge(self):
        files_paths = self.merge_paths_list.toPlainText().split("\n")
        self.merge_start_progressBar.setMaximum(len(files_paths)-1)
        destination_path = self.merge_destination_path.text().strip()

        # Check if two or more files are selected
        if len(files_paths) < 2:
            info = "Merge error"
            description = "Not enough source files selected, please select two or more files"
            self.display_message(info, description, "Error")
            # exit
            return
        # if

        # Initializing progrress counter
        prog_count = 0

        # Progress bar
        self.merge_start_button.hide()
        self.merge_start_progressBar.show()
        self.merge_start_progressBar.setValue(prog_count)

        try:
            # Merging selected files
            data_1 = pd.read_csv(files_paths[0])
            data_2 = pd.read_csv(files_paths[1])
            to_print = data_1.merge(data_2).set_index("Sample")
            to_print.to_csv(destination_path)
            
            # Increasing status bar
            prog_count += 1
            self.merge_start_progressBar.setValue(prog_count)

            for file_path in files_paths[2:]:
                data_1 = pd.read_csv(destination_path)
                data_2 = pd.read_csv(file_path)
                to_print = data_1.merge(data_2).set_index("Sample")
                to_print.to_csv(destination_path)

                # Increasing status bar
                prog_count += 1
                self.merge_start_progressBar.setValue(prog_count)
            # for
        except Exception as e:
            info = "Merge Error"
            description = str(e)
            self.display_message(info, description, "Error")
            self.merge_start_progressBar.hide()
            self.merge_start_button.show()
            # exit
            return
        # except

        info = "Operation Completed"
        description = "File have been created succesfully in " + destination_path
        self.display_message(info, description, "Success")

        # Clearing input fields
        self.merge_paths_list.setText("")
        self.merge_destination_path.setText("")
        self.merge_start_progressBar.hide()
        self.merge_start_button.show()
    # def start_merge

    # Apply settings in settings tab
    def apply_settings(self):
        try:
            cw = int(self.settings_classify_window_len_lineEdit.text().replace(" ", ""))
            pw = int(self.settings_predict_window_len_lineEdit.text().replace(" ", ""))
            co = int(self.settings_classify_overlap_lineEdit.text().replace(" ", ""))
            po = int(self.settings_predict_overlap_lineEdit.text().replace(" ", ""))
            # cs = int(self.settings_classify_samplerate_lineEdit.text().replace(" ", ""))
            # ps = int(self.settings_predict_samplerate_lineEdit.text().replace(" ", ""))
        except Exception as e:
            info = "Apllication Error"
            description = "An error occured trying to apply settings\n" + str(e)
            self.display_message(info,  description, "Error")
            # exit
            return
        # except

        compare_settings = True
        # Comparing checkboxes and lineEdit texts
        if not self.settings_classify_mean_checkbox.isChecked() == self.settings_predict_mean_checkbox.isChecked():
            compare_settings = False
        elif not self.settings_classify_variance_checkbox.isChecked() == self.settings_predict_variance_checkbox.isChecked():
            compare_settings = False
        elif not self.settings_classify_1_deg_der_mean_checkbox.isChecked() == self.settings_predict_1_deg_der_mean_checkbox.isChecked():
            compare_settings = False
        elif not self.settings_classify_2_deg_der_mean_checkbox.isChecked() == self.settings_predict_2_deg_der_mean_checkbox.isChecked():
            compare_settings = False
        elif not cw == pw:
            compare_settings = False
        elif not co == po:
            compare_settings = False
        # elif not cs == ps:
        #     compare_settings = False
        # if
    
        if not compare_settings:
            info = "Use comparable values"
            description = "Be careful to have the same settings when you classify and predict, it may cause unwanted behaviours"
            self.display_message(info, description, "Warning")
        # if

        # If the overlap value is greather than the window lenght give error
        # this will be an error, there will be no overlap if the sample is subdivided in sub_samples
        # having lenght of x and then asking to look for the next sample after y samples having y>=x
        # there will be no overlap
        if co >= cw or po >= pw:
            info = "Validation Error"
            description = "Please change the value of the overlap fields to be less than the window length"
            self.display_message(info, description, "Error")
            # exit
            return
        # if

        # Enabling reset button and disabling apply button
        self.settings_apply_button.setDisabled(True)
        self.settings_reset_button.setEnabled(True)
    # def apply_settings

    # Reset settings to default values
    def reset_settings(self):
        # Checkboxes
        # Mean
        self.settings_classify_mean_checkbox.setChecked(default_checkbox()) # True
        # Variance
        self.settings_classify_variance_checkbox.setChecked(default_checkbox()) # True
        # Mean der 1
        self.settings_classify_1_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Mean der 2
        self.settings_classify_2_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Window lenght
        self.settings_classify_window_len_lineEdit.setText(str(default_window_len()))
        # Window overlap
        self.settings_classify_overlap_lineEdit.setText(str(default_overlap()))
        # Samplerate
        # self.settings_classify_samplerate_lineEdit.setText(str(default_samplerate()))
        # Predict section
        # Mean
        self.settings_predict_mean_checkbox.setChecked(default_checkbox()) # True
        # Variance
        self.settings_predict_variance_checkbox.setChecked(default_checkbox()) # True
        # Mean der 1
        self.settings_predict_1_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Mean der 2
        self.settings_predict_2_deg_der_mean_checkbox.setChecked(default_checkbox()) # True
        # Window lenght
        self.settings_predict_window_len_lineEdit.setText(str(default_window_len()))
        # Window overlap
        self.settings_predict_overlap_lineEdit.setText(str(default_overlap()))
        # Samplerate
        # self.settings_predict_samplerate_lineEdit.setText(str(default_samplerate()))
        
        # Disabling reset button
        self.settings_reset_button.setDisabled(True)
        self.settings_apply_button.setDisabled(True)
    # def reset_settings

    # When settings are applyed, "apply" and "reset" buttons are enabled
    def settings_buttons_toggle(self):
        self.settings_reset_button.setEnabled(True)
        self.settings_apply_button.setEnabled(True)
    # def settings_buttons_toggle
# class MainWindow
