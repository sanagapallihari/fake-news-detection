from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM
from sklearn.preprocessing import OneHotEncoder
import keras.layers
from keras.models import model_from_json
import pickle
import os
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM

main = Tk()
main.title("DETECTION OF FAKE NEWS THROUGH IMPLEMENTATION OF DATA SCIENCE APPLICATION")
main.geometry("1300x1200")

global filename
X = None
Y = None
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer
global accuracy,error


# Ensure NLTK stopwords and wordnet are downloaded
import nltk
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    # Try using lemmatizer to trigger wordnet lookup
    _ = lemmatizer.lemmatize('test')
except LookupError:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier


def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="TwitterNewsData")
    textdata.clear()
    labels.clear()
    dataset = pd.read_csv(filename)
    dataset = dataset.fillna(' ')
    for i in range(len(dataset)):
        msg = dataset._get_value(i, 'text')
        label = dataset._get_value(i, 'target')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(int(label))
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+" ==== "+str(label)+"\n")
    


def preprocess():
    text.delete('1.0', END)
    global X, Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,2),smooth_idf=False, norm=None, decode_error='replace', max_features=200)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names_out())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    X = normalize(X)
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = Y.reshape(-1, 1)
    print(X.shape)
    encoder = OneHotEncoder()
    #Y = encoder.fit_transform(Y)
    # Ensure X is 3D for LSTM: (samples, timesteps, features)
    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    print(Y)
    print(Y.shape)
    print(X.shape)
    # Split and ensure train/test are also 3D
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    if tfidf_X_train.ndim == 2:
        tfidf_X_train = tfidf_X_train.reshape((tfidf_X_train.shape[0], tfidf_X_train.shape[1], 1))
    if tfidf_X_test.ndim == 2:
        tfidf_X_test = tfidf_X_test.reshape((tfidf_X_test.shape[0], tfidf_X_test.shape[1], 1))
    text.insert(END,"\n\nTotal News found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"\n")


import threading
def runLSTM():
    from keras.callbacks import Callback

    class TextBoxLogger(Callback):
        def __init__(self, text_widget):
            super().__init__()
            self.text_widget = text_widget
        def on_epoch_end(self, epoch, logs=None):
            msg = f"Epoch {epoch+1} completed. Accuracy: {logs.get('accuracy', 0):.4f}, Loss: {logs.get('loss', 0):.4f}\n"
            self.text_widget.insert('end', msg)
            self.text_widget.see('end')
    def train_lstm_thread():
        text.delete('1.0', END)
        global classifier, X, Y
        def train_and_save_model():
            from keras.layers import Input
            lstm_model = Sequential()
            lstm_model.add(Input(shape=(X.shape[1], X.shape[2])))
            lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(LSTM(128, activation='relu'))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(32, activation='relu'))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(2, activation='softmax'))
            lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            logger = TextBoxLogger(text)
            hist = lstm_model.fit(X, Y, epochs=10, validation_data=(tfidf_X_test, tfidf_y_test), callbacks=[logger], verbose=1)
            classifier = lstm_model
            classifier.save_weights('model/model_weights.weights.h5')
            model_json = classifier.to_json()
            with open("model/model.json", "w") as json_file:
                json_file.write(model_json)
            accuracy = hist.history
            with open('model/history.pckl', 'wb') as f:
                pickle.dump(accuracy, f)
            acc = accuracy['accuracy']                
            acc = acc[9] * 100
            text.insert(END,"\nLSTM Accuracy : "+str(acc)+"\n\n")
            text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')
            print(classifier.summary())
            return classifier

        if os.path.exists('model/model.json'):
            try:
                with open('model/model.json', "r") as json_file:
                    loaded_model_json = json_file.read()
                    classifier = model_from_json(loaded_model_json)
                classifier.load_weights("model/model_weights.weights.h5")
                print(classifier.summary())
                f = open('model/history.pckl', 'rb')
                data = pickle.load(f)
                f.close()
                acc = data['accuracy']
                acc = acc[9] * 100
                text.insert(END,"LSTM Fake News Detection Accuracy : "+str(acc)+"\n\n")
                text.insert(END,'LSTM Model Summary can be seen in black console for layer details\n')
            except Exception as e:
                text.insert(END, f"Model loading failed due to version incompatibility. Deleting old model files and retraining...\nError: {e}\n")
                import glob
                for file in glob.glob('model/model.*'):
                    try:
                        os.remove(file)
                    except Exception as remove_err:
                        text.insert(END, f"Could not delete {file}: {remove_err}\n")
                for file in glob.glob('model/history.pckl'):
                    try:
                        os.remove(file)
                    except Exception as remove_err:
                        text.insert(END, f"Could not delete {file}: {remove_err}\n")
                classifier = train_and_save_model()
        else:
            classifier = train_and_save_model()

    threading.Thread(target=train_lstm_thread).start()
        

    
def graph():
    try:
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
    except FileNotFoundError:
        messagebox.showerror("Error", "Model history file not found. Please train the model first.")
        return
    acc = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy','Loss'], loc='upper left')
    plt.title('LSTM Model Accuracy & Loss Graph')
    plt.show()

def predict():
    testfile = filedialog.askopenfilename(initialdir="TwitterNewsData")
    testData = pd.read_csv(testfile)
    text.delete('1.0', END)
    testData = testData.values
    testData = testData[:,0]
    print(testData)
    for i in range(len(testData)):
        msg = testData[i]
        msg1 = testData[i]
        print(msg)
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict = classifier.predict(testReview)
        print(predict)
        pred_class = np.argmax(predict)
        if pred_class == 0:
            text.insert(END,msg1+" === Given news predicted as GENUINE\n\n")
        else:
            text.insert(END,msg1+" == Given news predicted as FAKE\n\n")
        
    
font = ('times', 15, 'bold')
title = Label(main, text='DETECTION OF FAKE NEWS THROUGH IMPLEMENTATION OF DATA SCIENCE APPLICATION')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Fake News Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

dtButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
dtButton.place(x=20,y=200)
dtButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Test News Detection", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
