import numpy as np
import pandas as pd

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# load the data 

path = input("please give the path of your data file(csv): ")
df = pd.read_csv(path)

print(df.columns)

label = input("please select the label column: ")

Features = df.loc[:, df.columns != label]
labels =  df.loc[:, label]


X_train, X_test, y_train, y_test = train_test_split(Features,
                                                    labels,
                                                    test_size=0.2,
                                                   random_state = 5)

def regressor():
    models = []
    metrix = []
    train_accuracy = []
    test_accuracy = []
    models.append(('LinearRegression', LinearRegression()))
    models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
    models.append(('RandomForestRegressor', RandomForestRegressor()))
    models.append(('BaggingRegressor', BaggingRegressor()))
    models.append(('GradientBoostingRegressor', GradientBoostingRegressor()))
    models.append(('AdaBoostRegressor', AdaBoostRegressor()))
    models.append(('SVR', SVR()))
    models.append(('KNeighborsRegressor', KNeighborsRegressor()))
    for name, model in models:
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            r_square = r2_score(y_test,y_pred)
            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
            #print(name," ( r_square , rmse) is: ", r_square, rmse)
            metrix.append((name, r_square, rmse))
    print("""R_square - The Higher the R-squared, the better the model. ( R2 corresponds to the squared correlation between the observed outcome values and the predicted values by the model)

Root mean squared error is used to get the accuracy of model: lowest RMSE - the best one in your case (MSE = mean((observeds - predicteds)^2) and RMSE = sqrt(MSE))

Residual Standard Error (RSE) -  The lower the RSE, the better the model. In practice, the difference between RMSE and RSE is very small, particularly for large multivariate data.

Mean Absolute Error (MAE) -  MAE = mean(abs(observeds - predicteds)). MAE is less sensitive to outliers compared to RMSE.   """)
    return metrix

def classifier():
    models = []
    metrix = []
    c_report = []
    train_accuracy = []
    test_accuracy = []
    models.append(('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('GaussianNB', GaussianNB()))
    models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=100)))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('Linear_SVM', LinearSVC()))
    models.append(('XGB', XGBClassifier()))
    models.append(('SGD', SGDClassifier()))
    models.append(('Perceptron', Perceptron()))
    for name, model in models:
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            train_acc = round(m.score(X_train, y_train) * 100, 2)
            test_acc = metrics.accuracy_score(y_test,y_pred) *100
            c_report.append(classification_report(y_test, y_pred))
            #print(name," (train accuracy , test_accuracy) is: ", train_acc, test_acc)
            metrix.append([name, train_acc, test_acc])
    print("model.score = how well the model is trained (training accuracy)")
    print("accuracy_score = how well the model is predicted (testing accuracy)")
    print("confusion matrix to evaluate performance of data")
    return metrix

if(len(labels.unique())<10):
    metrix = classifier()
    for i in metrix:
        print(i)
else:
    metrix = regressor()
    for i in metrix:
        print(i)