
# To ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



##############################################

import numpy as np
import pandas as pd


##############Regressors#####################

#mainmodels
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor



#other models
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor

from sklearn.compose import TransformedTargetRegressor

from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR

from sklearn.neural_network import MLPRegressor

from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression

from sklearn.tree import ExtraTreeRegressor

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.isotonic import IsotonicRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.multioutput import MultiOutputRegressor

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.multioutput import RegressorChain




##############################################

###################Metricc#########################

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

######################################################


###################Classifiers#################################

#Main models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#OtherModels


from sklearn.tree import ExtraTreeClassifier

from sklearn.svm.classes import OneClassSVM
from sklearn.svm import NuSVC

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from sklearn.neighbors.classification import RadiusNeighborsClassifier

from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier

from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    

from sklearn.gaussian_process.gpc import GaussianProcessClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB  


from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV


from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

from sklearn.neighbors import NearestCentroid

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


##################################################################

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

def default_regressor_models():
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
    test_acc=[]
    names=[]
    for name, model in models:
        try:
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            r_square = r2_score(y_test,y_pred)
            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
            test_acc.append(r_square)
            names.append(name)            
            #print(name," ( r_square , rmse) is: ", r_square, rmse)
            metrix.append((name, r_square, rmse))
        except:
            print("Excepton Occured  : ",name)
    return metrix,test_acc,names


def all_regressor_models():
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
    models.append(('ARDRegression', ARDRegression()))
    models.append(('BayesianRidge', BayesianRidge()))
    models.append(('ElasticNet', ElasticNet()))
    models.append(('ElasticNetCV', ElasticNetCV()))
    models.append(('Lars', Lars()))
    models.append(('LassoCV', LassoCV()))
    models.append(('LassoLars', LassoLars()))
    models.append(('LassoLarsCV', LassoLarsCV()))
    models.append(('MultiTaskElasticNet', MultiTaskElasticNet()))
    models.append(('MultiTaskLasso', MultiTaskLasso()))
    models.append(('MultiTaskLassoCV', MultiTaskLassoCV()))
    models.append(('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit()))
    models.append(('OrthogonalMatchingPursuitCV', OrthogonalMatchingPursuitCV()))
    models.append(('PassiveAggressiveClassifier', PassiveAggressiveClassifier()))
    models.append(('RANSACRegressor', RANSACRegressor()))
    models.append(('Ridge', Ridge()))
    models.append(('RidgeCV', RidgeCV()))
    models.append(('SGDRegressor', SGDRegressor()))
    models.append(('TheilSenRegressor', TheilSenRegressor()))
    models.append(('TransformedTargetRegressor', TransformedTargetRegressor()))
    models.append(('LinearSVR', LinearSVR()))
    models.append(('NuSVR', NuSVR()))
    models.append(('MLPRegressor', MLPRegressor()))
    models.append(('CCA', CCA()))
    models.append(('PLSRegression', PLSRegression()))
    models.append(('PLSCanonical', PLSCanonical()))
    models.append(('GaussianProcessClassifier', GaussianProcessClassifier()))
    models.append(('GradientBoostingRegressor', GradientBoostingRegressor()))
    models.append(('HistGradientBoostingRegressor', HistGradientBoostingRegressor()))
    estimators = [('lr', RidgeCV()),('svr', LinearSVR(random_state=42))]
    models.append(('StackingRegressor', StackingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=10,random_state=42))))
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    models.append(('VotingRegressor', VotingRegressor([('lr', r1), ('rf', r2)])))
    models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
    models.append(('IsotonicRegression', IsotonicRegression()))
    models.append(('KernelRidge', KernelRidge()))
    models.append(('RadiusNeighborsClassifier', RadiusNeighborsClassifier()))
    test_acc=[]
    names=[]
    for name, model in models:
        try:
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            r_square = r2_score(y_test,y_pred)
            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
            test_acc.append(r_square)
            names.append(name)            
            #print(name," ( r_square , rmse) is: ", r_square, rmse)
            metrix.append((name, r_square, rmse))
        except:
            print("Excepton Occured  : ",name)
    return metrix,test_acc,names

def default_classifier_models():
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
    test_accuracy= []
    names = []
    for name, model in models:
        try:
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            train_acc = round(m.score(X_train, y_train) * 100, 2)
            test_acc = metrics.accuracy_score(y_test,y_pred) *100
            c_report.append(classification_report(y_test, y_pred))
            test_accuracy.append(test_acc)
            names.append(name)
            metrix.append([name, train_acc, test_acc])
        except:
            print("Exception Occurred  :",name)
    return metrix,test_accuracy,names

def all_classifier_models():
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
    models.append(('ExtraTreeClassifier', ExtraTreeClassifier()))
    models.append(('OneClassSVM', OneClassSVM(gamma = 'auto')))
    models.append(('NuSVC', NuSVC()))
    models.append(('MLPClassifier', MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)))
    models.append(('RadiusNeighborsClassifier', RadiusNeighborsClassifier(radius=2.0)))
    models.append(('OutputCodeClassifier', OutputCodeClassifier(estimator=RandomForestClassifier(random_state=0),random_state=0)))
    models.append(('OneVsOneClassifier', OneVsOneClassifier(estimator = RandomForestClassifier(random_state=1))))
    models.append(('OneVsRestClassifier', OneVsRestClassifier(estimator = RandomForestClassifier(random_state=1))))
    models.append(('LogisticRegressionCV', LogisticRegressionCV()))
    models.append(('RidgeClassifierCV', RidgeClassifierCV()))
    models.append(('RidgeClassifier', RidgeClassifier()))
    models.append(('PassiveAggressiveClassifier', PassiveAggressiveClassifier()))
    models.append(('GaussianProcessClassifier', GaussianProcessClassifier()))
    models.append(('HistGradientBoostingClassifier', HistGradientBoostingClassifier()))
    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))]
    models.append(('StackingClassifier', StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())))
    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    models.append(('VotingClassifier', VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
    models.append(('BaggingClassifier', BaggingClassifier()))
    models.append(('ExtraTreesClassifier', ExtraTreesClassifier()))
    models.append(('CategoricalNB', CategoricalNB()))
    models.append(('ComplementNB', ComplementNB()))
    models.append(('BernoulliNB', BernoulliNB()))
    models.append(('MultinomialNB', MultinomialNB()))
    models.append(('CalibratedClassifierCV', CalibratedClassifierCV()))
    models.append(('LabelPropagation', LabelPropagation()))
    models.append(('LabelSpreading', LabelSpreading()))
    models.append(('NearestCentroid', NearestCentroid()))
    models.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()))
    models.append(('GaussianMixture', GaussianMixture()))
    models.append(('BayesianGaussianMixture', BayesianGaussianMixture()))
    
    test_accuracy= []
    names = []
    for name, model in models:
        try:
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            train_acc = round(m.score(X_train, y_train) * 100, 2)
            test_acc = metrics.accuracy_score(y_test,y_pred) *100
            c_report.append(classification_report(y_test, y_pred))
            test_accuracy.append(test_acc)
            names.append(name)
            metrix.append([name, train_acc, test_acc])
        except:
            print("Exception Occurred  :",name)
    return metrix,test_accuracy,names


########################################################################

### Determine classification/Regression

if(len(labels.unique())<20):
    print("Press 1 for running default models\n2.Press 2 for running all models\n")
    choice = input()
    metrix = []
    if(choice == '1'):
        metrix,test_acc,model_names = default_classifier_models()
    elif(choice == '2'):
        metrix,test_acc,model_names = all_classifier_models()
    print("\n\n\n######################################\n\t\t\t")
    mapped = zip(test_acc,model_names)
    mapped = sorted(mapped,reverse=True)
    print("\n\t\t\t")
    print(mapped[0])
    print("\n\t\t\t")
    print(mapped[1])
    print("\n\t\t\t")
    print(mapped[2])
    print("\n\t\t\t")
    print("\n\n\n######################################\n")
    print("Do You want the output of almodel?(Press y/n)")
    choice = input()
    if(choice == 'y'):
        for i in metrix:
            print(i)
else:
    print("Press 1 for running default models\n2.Press 2 for running all models\n")
    choice = input()
    metrix = []
    if(choice == '1'):
        metrix,test_acc,model_names = default_regressor_models()
    elif(choice == '2'):
        metrix,test_acc,model_names = all_regressor_models()
    print("\n\n\n######################################\n\t\t\t")
    mapped = zip(test_acc,model_names)
    mapped = sorted(mapped,reverse=True)
    print("\n\t\t\t")
    print(mapped[0])
    print("\n\t\t\t")
    print(mapped[1])
    print("\n\t\t\t")
    print(mapped[2])
    print("\n\t\t\t")
    print("\n\n\n######################################\n")
    print("Do You want the output of almodel?(Press y/n)")
    choice = input()
    if(choice == 'y'):
        for i in metrix:
            print(i)


s
