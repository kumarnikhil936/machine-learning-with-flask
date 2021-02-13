# Flask Packages
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, ALL
from flask_sqlalchemy import SQLAlchemy

from werkzeug.utils import secure_filename
import os
import datetime
import time

# EDA Packages
import pandas as pd

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

app = Flask(__name__)
Bootstrap(app)
# db = SQLAlchemy(app)
#
# # configuration for file uploads
# files = UploadSet('files', ALL)
# app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
# configure_uploads(app, files)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'


# Saving Data To Database Storage
# class FileContents(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(300))
#     modeldata = db.Column(db.String(300))
#     data = db.Column(db.LargeBinary)


@app.route('/')
def index():
    return render_template('index.html')


# Route for the Processing and Details Page
@app.route('/dataupload', methods=['GET', 'POST'])
def dataupload():
    if request.method == 'POST' and 'csv_data' in request.files:
        # Get the file and the filename
        file = request.files['csv_data']
        filename = secure_filename(file.filename)

        # Save file in the uploadsDB folder
        file.save(os.path.join('static', 'uploadsDB', filename))
        full_filename = os.path.join('static', 'uploadsDB', filename)

        # For Time
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # Perform EDA
        df = pd.read_csv(full_filename)
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns)
        df_target_name = df[df.columns[-1]].name
        df_feature_names = df_columns[0:-1]
        df_X = df.iloc[:, 0:-1]
        df_Y = df.iloc[:, -1]

        # Model Building
        models = list()
        models.append(('LogisticRegression', LogisticRegression()))
        models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
        models.append(('KNeighborsClassifier', KNeighborsClassifier()))
        models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
        models.append(('GaussianNB', GaussianNB()))
        models.append(('SupportVectorClassifier', SVC()))

        # evaluate each model in turn
        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'

        for name, model in models:
            try:
                names.append(name)
                kfold = model_selection.KFold(n_splits=10)

                # continuous values are not supported
                df_X = df_X.round()
                df_Y = df_Y.round()

                cv_results = model_selection.cross_val_score(model, df_X, df_Y, cv=kfold, scoring=scoring)
                results.append(cv_results)
                msg = "%s | %f | %f" % (name, cv_results.mean(), cv_results.std())
                allmodels.append(msg)
            except:
                pass

        # Saving Results of Uploaded Files  to Sqlite DB
        # newfile = FileContents(name=file.filename, data=file.read(), modeldata=msg)
        # db.session.add(newfile)
        # db.session.commit()

    return render_template('details.html', filename=filename, date=date,
                           df_size=df_size,
                           df_shape=df_shape,
                           df_feature_names=df_feature_names,
                           df_target_name=df_target_name,
                           model_results=allmodels,
                           model_names=names,
                           fullfile=full_filename,
                           dfplot=df[:20]
                           )


if __name__ == '__main__':
    app.run(debug=True)
