import pandas as pd  
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt  

from sklearn_pandas import DataFrameMapper
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix


orig_dataset=pd.read_excel('BreastCancer_Prognostic_v1.xlsx',na_values = "?", sep=",")
dataset = orig_dataset.copy()

dataset.isna().sum()
'''
Checking the presence of NA values and dropping them we use dataset.isna().sum()
'''

dataset = dataset.dropna()
y = dataset['Outcome']
dropped_params = ['ID', 'Time', 'Outcome']
dataset = dataset.drop(dropped_params, 1)

dataset.describe()
'''
to view statistics related to data
'''

plot = sns.countplot(y,label="Count")
N, R = y.value_counts()
print('Total Non-Recurring Cases are',N)
print('Total Recurring Cases are', R)


dataset_without_fe = dataset

fig = plt.subplots(figsize = (32, 32))
sns.set(font_scale=1.6)
sns.heatmap(dataset.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
plt.savefig("Heat_Map_for_predicting_breast_cancer_outcome.png")


dropped_params = ['texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','symmetry_mean',
            'radius_std_dev','texture_std_dev','perimeter_std_dev','area_std_dev','smoothness_std_dev','compactness_std_dev',
             'concavity_std_dev','concave_points_std_dev','symmetry_std_dev',
             'Worst_texture','Worst_perimeter','Worst_area','Worst_smoothness','Worst_compactness',
             'Worst_concavity','Worst_concave_points','Worst_symmetry','Tumor_Size','Lymph_Node_Status']

'''
These Dropped parameters are highly correlated variables because
it could introduce a problem of multicollinearity which further has a negative impact on the accuracy of the model.
'''

featureEngineered_dataset = dataset.drop(dropped_params,axis = 1 )
featureEngineered_dataset.head()

mapper = DataFrameMapper([(featureEngineered_dataset.columns, StandardScaler())])
scaled_features = mapper.fit_transform(featureEngineered_dataset.copy(), 4)
scaled_features_df = pd.DataFrame(scaled_features, index=featureEngineered_dataset.index, columns=featureEngineered_dataset.columns)
'''
scaled_features_df is the dataset on which feaured engineering 
has been performed
'''

scaled_features_df.describe()

i=4
def running_and_evaluating_model(x, y):
    
    global i
    i=i+1
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    regr = linear_model.LogisticRegression(solver = "lbfgs", max_iter = 3000)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    
    df=pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
    s="Heat_Map_for_predicting_breast_cancer_outcome"+str(i)+".xlsx"
    df.to_excel(s)

    accuracy = regr.score(x_test, y_test)
    print("Accuracy: " , accuracy * 100)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,annot=True,fmt="d")
    

running_and_evaluati2ng_model(dataset_without_fe, y)

running_and_evaluating_model(scaled_features_df, y)

    
