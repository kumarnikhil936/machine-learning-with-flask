from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import os.path

app = Flask(__name__)
saved_model = 'trained_model.sav'
saved_countvec = 'count_vectorizer.sav'


def train_model():
    # check if a trained model is already present in the folder
    if os.path.isfile(saved_model) & os.path.isfile(saved_countvec):
        print('No training required! Trained model and count vectorizer already present in the folder.')
    else:
        # combine all the files together
        df = pd.DataFrame()
        for filename in ['Eminem.csv', 'KatyPerry.csv', 'LMFAO.csv', 'Psy.csv', 'Shakira.csv']:
            df_temp = pd.read_csv(f'data/{filename}')
            df = df.append(df_temp)
        df = df[['CONTENT', 'CLASS']]

        # define features and labels
        df_x = df['CONTENT']
        df_y = df['CLASS']

        # extract features with CountVectorizer
        corpus = df_x
        cvec = CountVectorizer()
        trans_x = cvec.fit_transform(corpus)

        # split train and test data
        train_x, test_x, train_y, test_y = train_test_split(trans_x, df_y, test_size=0.2, random_state=42)

        # define, train, and evaluate classifier
        clf = MultinomialNB()
        clf.fit(train_x, train_y)
        print(clf.score(test_x, test_y))

        # save model and count_vectorizer
        pickle.dump(clf, open(saved_model, 'wb'))
        pickle.dump(cvec, open(saved_countvec, 'wb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # train the model if a trained model is already not present
        train_model()

        # load model and count_vectorizer from the pickled files
        clf = pickle.load(open(saved_model, 'rb'))
        cvec = pickle.load(open(saved_countvec, 'rb'))

        # get data from request
        comment = request.form['comment']
        data = [comment]

        # make predictions
        vect = cvec.transform(data).toarray()
        pred = clf.predict(vect)

    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
