# Neural-Machine-Translation-System
Neural Machine Translation System is written in Keras(version 2 or higher) and python3 Scipy environment to translate 
German phrases to English.<br>
**Dataset**-Dataset is taken from http://www.manythings.org/ website and is then further cleaned and processed.Remove all non-printable characters.
Data cleaning is done to remove all punctuation characters and to normalize all unicode characters to ASCII ,lowercase and to remain 
non-alphabetic characters.
Pickling is used to serialize the data.An encoder-decoder model LSTM model is being used in the model and is trained using the 
Adam's approach and is minimised using the categorical loss function.
The approach and code snippet are inspired from https://machinelearningmastery.com.
