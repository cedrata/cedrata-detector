# ###########################################################################
# Author: Luca Greggio
# Github: ...
# Email: luca01.greggio@edu.unife.it, greggius9891@gmail.com
# This is a machine learning project able to detect where notes are played 
# in audio tracks. 
# If you are interested in it please please beofre using this code contact me
# ###########################################################################

# This file containg the classes used to apply classification and prediction

# python modules
import os
import csv
import numpy as np

# librosa module
import librosa

# Weka related modules
from weka.classifiers import Classifier
from weka.core.converters import Loader

# Parent class of Classification and Prediction
class Extract():
    # mean: an empty list filled with the mean of each sub-sample
    # var: an empty list filled with the variance of each sub-sample
    # mean_der_1: an empty list filled with the 1st degree derivative of the mean
    # mean_der_2: an empty list filled with the 2nd degree derivative of the mean
    # audio_ext: are the accepted audio formats
    def __init__(self):
        self._mean = []
        self._var = []
        self._mean_der_1 = []
        self._mean_der_2 = []
        self._note_class = []
        self._audio_ext = [".wav", ".aiff", ".aif"]
    # def __init__


    # Calculaitng the needed audio features given a source
    # smean, svar, s1der, s2der: bool values that specify what to calculate
    # smean is standing for "settings mean"
    # **args is used to print a fixed char in self._note_class
    # e.g. if features are calculated to do a prediction
    # the note_class value (1, 2, 3, 4, 5) will be unkown so must be "?"
    def calc_features(self, source: str, smean: bool, svar: bool, s1der: bool, s2der: bool, win_len: int, overlap: int, *args):
        if source == "":
            raise Exception ("Please fill all the fields")
        # if
        if not (os.path.isfile(source) or source.endswith(tuple(self._audio_ext))):
            raise Exception("The audio source path must be a file ending with .wav .aif .aiff ")
        # if
        # importing audio data and samplerate given path
        # sr=None to preserve original sample rate
        # if the sample rate is higher than 44100 resample!
        data, samplerate = librosa.load(source, sr=None)
        if samplerate > 44100: 
            data = librosa.resample(data, samplerate, 44100)
        # if
        # min_data_len = int(2*samplerate/10) # is the minimum sub-sample and accepted length
        min_data_len = win_len # is the minimum sub-sample and accepted length
        
        # The sample has not enough points of measuerement
        # return the filepath string to print a warning or log info
        if len(data) < min_data_len:
            return source
        # if

        # defining sub sample starting and ending point
        sub_sample_start = 0
        sub_sample_end = min_data_len
        # sub sample will be shifted 0.5 times the min_data_len
        # If sub_sample_start = 0 and sub_sample_end = min_data_len = 8820
        # then sub_sample_shift = 4410, new sub_sample will start from 0+4410, end in 8820+4410
        # sub_sample_shift = int(min_data_len/2)  
        sub_sample_shift = overlap
        # Every sub sample will be classified with a value in range(1,10)
        # value 10 means this is certanly a the begin of a note
        # value 1 is not at all the begin of a note
        sub_sample_value = 5

        # While the sample as enough points of measurement it'll loop
        while sub_sample_end < len(data):

            # Selecting alll samples from the 1st to the 8820st
            sub_sample = data[sub_sample_start:sub_sample_end]
            # Calculating and writing mean
            if smean is True: 
                self._mean.append(np.mean(sub_sample))
            # Caluclating and writing variance
            if svar is True: 
                self._var.append(np.var(sub_sample))
            # Calculating and writing 1st degree mean derivative
            if s1der is True: 
                self._mean_der_1.append(np.mean(np.polyder(sub_sample, 1)))
            # Calculating and writing 2nd degree mean derivative
            if s2der is True: 
                self._mean_der_2.append(np.mean(np.polyder(sub_sample, 2)))
            # If a value is specified (as it can be specified in prediction)
            # this value will be added in the note_class list
            if len(args):
                if args[0] == "?":
                    self._note_class.append(args[0])
            elif sub_sample_value < 1:
                self._note_class.append("1")
            else:
                self._note_class.append(str(sub_sample_value))
                sub_sample_value-=1
            # if

            # Selecting all samples from the i-st to the i+8820-st
            sub_sample_start+=sub_sample_shift
            sub_sample_end+=sub_sample_shift
        # while
    # def calc_features

    # Normalizing features to be able to compare them before applying classification and prediction
    def norm_features(self, smean: bool, svar: bool, s1der: bool, s2der: bool):
        normalize = lambda current, feature: (current - min(feature)) / (max(feature) - min(feature))
        # norm_mean = []
        # norm_var = []
        # norm_mean_der_1 = []
        # norm_mean_der_2 = []
        # for i, (mean, var, mean_der_1, mean_der_2) in enumerate(itertools.zip_longest(self._mean, self._var, self._mean_der_1, self._mean_der_2)):
        #     # normalizing mean
        #     if smean is True:
        #         norm_mean.append(normalize(mean, self._mean))                

        #     # normalizing variance
        #     if svar is True: 
        #         norm_var.append(normalize(var, self._var))

        #     # normalizing the mean 1st degree derivative
        #     if s1der is True: 
        #         norm_mean_der_1.append(normalize(mean_der_1, self._mean_der_1))

        #     # normalizing the mean 2nd degree derivative
        #     if s2der is True: 
        #         norm_mean_der_2.append(normalize(mean_der_2, self._mean_der_2))
        #     print(i)
        # # for
       
        if smean is True:
            norm_mean = []
            for i in range(len(self._mean)):
                print(i)
                norm_mean.append(normalize(self._mean[i], self._mean))
            # for
            self._mean = norm_mean
        # if

        if svar is True: 
            norm_var = []
            for i in range(len(self._var)):
                print(i)
                norm_var.append(normalize(self._var[i], self._var))
            # for
            self._var = norm_var
        # if

        if s1der is True: 
            norm_mean_der_1 = []
            for i in range(len(self._mean_der_1)):
                print(i)
                norm_mean_der_1.append(normalize(self._mean_der_1[i], self._mean_der_1))
            # for
            self._mean_der_1 =  norm_mean_der_1
        # if
        
        if s2der is True: 
            norm_mean_der_2 = []
            for i in range(len(self._mean_der_2)):
                print(i)
                norm_mean_der_2.append(normalize(self._mean_der_2[i], self._mean_der_2))
            # for
            self._mean_der_2 =  norm_mean_der_2
        # if

    # def norm_features

    # Prints arff file with calculated features
    # rname: is the rleation name, also the file name
    # destination: is the folder where to store the rname.arff file
    def save_arff(self, rname: str, destination: str, smean: bool, svar: bool, s1der: bool, s2der: bool):
        if rname == "" or destination == "":
            raise Exception ("Please fill all the fields")
        # if
        if not os.path.isdir(destination):
            raise Exception("The destination field must be a valid direcotry")
        # if
        try:
            # Generating *.arff path to store extracted features
            extension = "arff"
            arff_file = os.path.join(destination, rname + "." + extension)

            # Writing arff_file
            with open(arff_file, "w") as fp: 

                fp.write('@relation ' + rname + '\n')
                if smean is True: fp.write('@attribute mean numeric\n')
                if svar is True: fp.write('@attribute variance numeric\n')
                if s1der is True: fp.write('@attribute mean_der_1 numeric\n')
                if s2der is True: fp.write('@attribute mean_der_2 numeric\n')
                fp.write('@attribute note {1, 2, 3, 4, 5}\n\n')
                fp.write('@data\n')
                max_i = max(len(self._mean), len(self._var), len(self._mean_der_1), len(self._mean_der_2))
                i = 0
                for i in range(max_i):
                    if smean is True: 
                        fp.write("{0:1.4f}".format(self._mean[i]))
                    if svar is True: 
                        fp.write(",{0:1.4f}".format(self._var[i]))
                    if s1der is True: 
                        fp.write(",{0:1.4f}".format(self._mean_der_1[i]))
                    if s2der is True: 
                        fp.write(",{0:1.4f}".format(self._mean_der_2[i]))
                    fp.write(",{0:s}\n".format(self._note_class[i]))
                # for mean, var, mean_der_1, mean_der_2, note in zip(self._mean, self._var, self._mean_der_1, self._mean_der_2, self._note_class):
                #     fp.write("{0:1.4f},{1:1.4f},{2:1.4f},{3:1.4f},{4:s}\n".format(mean, var, mean_der_1, mean_der_2, note))
                # for
            # with
        except:
            raise Exception("Unexpected error during the creation of the arff file")
        # except
        return arff_file
    # def save_arff
# class Extract

# Chilren class from Extract
# used to classify samples to use in prediction
class Classification(Extract):

    # wav_files: an empty list filled with audio paths
    # wav_files_qty: the number of found audio files
    def __init__(self):
        self._wav_files = []
        self._wav_files_qty = 0
        Extract.__init__(self)
    # def __init__

    # Property methods
    
    @property
    def wav_files(self):
        return self._wav_files
    # def wav_files

    # End property methods

    # Find_audio finds all wav files in the current directory
    # append them in "wav_files"
    # source: is the folder where to search the files 
    def find_audio(self, source: str):
        if source == "":
            raise Exception ("Please fill all the fields")
        # if
        if not os.path.isdir(source):
            raise Exception("The source field must be a valid direcotry")
        # if
        try:
            ext = [".wav", ".aiff", ".aif"]
            for wav_file in os.listdir(source):
                if wav_file.endswith(tuple(self._audio_ext)):
                    self._wav_files.append(os.path.join(source, wav_file))
                    self._wav_files_qty += 1
                # if
            # for
        except: 
            raise OSError("Unexpected error during the reading of audio files")
        # except
        if self._wav_files_qty == 0:
            raise Exception("No audio files were found in the specified directory")
        # if
        self._wav_files.sort()
    # def find_audio

    # NOT NEEDED ANT THE MOMENT
    # # Create the model to use during predictions
    # # rname: is the rleation name, also the file name
    # # destination: is the folder where to store the rname.arff file
    # def save_model(self, rname: str, destination: str):
    #     # Generate the path to save *.model  file
    #     extension = "model"
    #     model_file = os.path.join(destination, rname + "." + extension)

    #     # Retriving the *.file path to create the *.model file
    #     extension = "arff"
    #     arff_file = os.path.join(destination, rname + "." + extension)

    #     # Loading data from *.arff
    #     loader = Loader(classname="weka.core.converters.ArffLoader")
    #     data = loader.load_file(arff_file)
    #     data.class_is_last()

    #     # Training
    #     cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
    #     cls.build_classifier(data)

    #     # Printing
    #     cls.serialize(model_file, header=data)
    # # def dave_model
# class Classification

# Children class from Extraction
# used to predict where notes will be in audio given a classification
class Prediction(Extract):

    # Predicted: is the value of each sample
    # the probability of it to be a note
    def __init__(self):
        self._predicted = []
        Extract.__init__(self)
    # def __init__

    # Strart the prediction
    # to_test_file: is the *.arff file containing features and the note_class values == "?"
    #               of the audio file where to apply prediction
    # trained_file: is the file containign the cllassified features, the source to build the classifier
    def predict(self, to_test_file: str, trained_file: str):
        if to_test_file == "" or trained_file == "":
            raise Exception ("Please fill all the fields")
        # if
        if not(os.path.isfile(to_test_file) or os.path.isfile(trained_file)):
            raise Exception ("The file to test and the trained file must be paths to existing files")
        # if
        if not (to_test_file.endswith(".arff") or trained_file.endswith(".arff")):
            raise Exception ("The file to test and the trained one must be arffs files")
        # if

        # Checking files headers, they must be the same to do predictions
        trained_header = ""
        to_test_header = ""
        try:
            with open(trained_file, "r") as tf:
                read_file = True
                while read_file:
                    line = tf.readline()
                    if not "@data" in line:
                        trained_header += line
                    else:
                        read_file = False
                    # if
                # while
            # with

            with open(to_test_file, "r") as tt:
                read_file = True
                while read_file:
                    line = tt.readline()
                    if not "@data" in line:
                        to_test_header += line
                    else:
                        read_file = False
                    # if
                # while
            # with
        except:
            raise Exception("Error opening the referencce arff file")
        # except

        if not trained_header == to_test_header:
            raise Exception("Files header must be the same")
        # if

        # Loading trained and test data from arff files
        loader = Loader(classname="weka.core.converters.ArffLoader")
        trained_data = loader.load_file(trained_file)
        trained_data.class_index = trained_data.num_attributes - 1
        to_test_data = loader.load_file(to_test_file)
        to_test_data.class_is_last()

        # Building classifier from trained data
        cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
        cls.build_classifier(trained_data)

        # Evaluating predictions
        for i, inst in enumerate(to_test_data):
            pred = cls.classify_instance(inst)
            # dist = cls.distribution_for_instance(inst)
            row = [int(i+1), int(pred+1)]
            self._predicted.append(row)
        # for
    # def predict

    # Create the csv where to save predictions
    # rname: is the rleation name, also the file name
    # destination: is the folder where to store the rname.arff file
    def save_prediction(self, rname: str, destination: str):
        if rname == "" or destination == "":
            raise Exception ("Please fill all the fields")
        if not os.path.isdir(destination):
            raise Exception("The destination field must be a valid direcotry")
        # Preparing data for csv
        fields = ["Sample", rname]

        # Printing csv file
        extension = "csv"
        prediction = os.path.join(destination, rname + "." + extension)

        # Create the model to use during predictions
        # rname: is the rleation name, also the file name
        # destination: is the folder where to store the rname.arff file
        try:
            with open(prediction, "w") as csvfile:
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 

                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(self._predicted)
            # with
        except:
            raise Exception("Unexpected error during the creation of the result file")
        # except
    # def save_prediction
# class Prediction