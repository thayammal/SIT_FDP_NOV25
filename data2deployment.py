import pandas as pd
data = pd.read_csv("placement_details.csv")

# Check data samples
print(data.head())

#Know about the features and target/label
feature_list = data.columns[:-1].values
label = [data.columns[-1]]
print('Features:\n',feature_list,'\n')
print('Target:\n',label)

# Data description 
print(data.info())


# # Numerical Data Statistics 
num_features = data.select_dtypes(include='number')
# print(num_features.describe())

# # #categorical data statitics
cat_features = data.select_dtypes(include='object')
# print(cat_features.describe())


# # #Graphical summaries
import seaborn as sns
import matplotlib.pyplot as plt

# #Histogram of numerical features
#num_features.hist(bins=50,figsize=(12,12))
 # display histogram
#plt.show()



# # #Categorical features : bar chart
# # for col in cat_features.columns:
# #     plt.figure(figsize=(8, 5))
# #     cat_features[col].value_counts().plot(kind='bar')
# #     plt.title(f"Category Counts for {col}")
# #     plt.xlabel(col)
# #     plt.ylabel("Count")
# #     plt.xticks(rotation=45, ha='right')
# #     plt.show()



# # #It's a good idea to create a copy of the training set so that we can freely
# # # manipulate it without worrying about any manipulation in the original set.

data_copy = data.copy()


# plt.grid(True)
# sns.scatterplot(x='IQ', y='CGPA', hue='Placement',
#                  data=data_copy)
# plt.title('IQ vs CGPA')
# plt.show()

# # # plt.grid(True)
# # # sns.scatterplot(x='Prev_Sem_Result', y='CGPA', hue='Placement',
# # #                 data=data_copy)
# # # plt.title('Prev_Sem_Result vs CGPA')
# # # plt.show()

# # plt.grid(True)
# # sns.scatterplot(x='Academic_Performance', y='Communication_Skills', hue='Placement',
# #                 data=data_copy)
# # plt.title('IQ vs CGPA')
# # plt.show()


import numpy as np

corr_matrix = num_features.corr()

plt.figure(figsize=(14,7))
mask = np.triu(corr_matrix)
sns.heatmap(corr_matrix, annot=True, mask= mask)
plt.show()

# #attribute_list = ['IQ', 'Prev_Sem_Result', 'Academic_Performance', 'Extra_Curricular_Score','Communication_Skills','Projects_Completed']
# # sns.pairplot(data_copy[attribute_list])
# # plt.show()


# Prepare dataset fro ML 
#4.1 Separate features and labels from the training set. 
# scikit-learn
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(data, data["Placement"]):
  strat_train_set = data.loc[train_index]
  strat_test_set = data.loc[test_index]
  print(strat_train_set.head(), strat_test_set.head())


#print(strat_train_set.describe())
#print(strat_test_set.describe())

attribute_list = ['IQ', 'Prev_Sem_Result', 'Academic_Performance', 'Extra_Curricular_Score','Communication_Skills','Projects_Completed']

sns.pairplot(strat_train_set[attribute_list])
plt.show()

sns.pairplot(strat_test_set[attribute_list])
plt.show()




# # # Copy all features leaving aside the label.
data_features = strat_train_set.drop("Placement", axis=1)

# # Copy the label list
data_labels = strat_train_set['Placement'].copy()

# print(data_features.head())
# print(data_labels.head())



# #4.2 Data cleaning
# #check the missing values
print(data_features.isna().sum())


from sklearn.impute import SimpleImputer   #scikit-learn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression , LinearRegression 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


num_features = data_features.select_dtypes(include='number')
cat_features = data_features.select_dtypes(include='object')
num_cols = list(num_features.columns) #[IQ, College_ID, Prev_Sem_Result, Academic_Performance, Extra_Curricular_Score, Communication_Skills, Projects_Completed]
cat_cols = list(cat_features.columns) #[]


# # Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
('imputer', SimpleImputer(strategy='most_frequent')),
('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# # Combine both pipelines
preprocessor = ColumnTransformer([
 ('num', num_pipeline, num_cols),
 ('cat', cat_pipeline, cat_cols)
 ])

# # Define the LogisticRegression Pipeline
full_pipeline = Pipeline([
     ('preprocessor', preprocessor),  # Assuming 'preprocessor' is defined
     ('classifier', LogisticRegression(random_state=42))   
 ])

lin_reg_model = full_pipeline.fit(data_features, data_labels) 



# #Testing
# # Separate features and labels in test set
X_test = strat_test_set.drop("Placement", axis=1)
y_test = strat_test_set["Placement"].copy()

# # # Predict labels for test set
y_pred = lin_reg_model.predict(X_test)

# # # Calculate accuracy
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

acc = accuracy_score(y_test, y_pred)
# print("Test set accuracy:", acc)

# # # # More detailed metrics
# # # print(classification_report(y_test, y_pred))
# # # print(confusion_matrix(y_test, y_pred))


#save the model .h5. pkl
import joblib
joblib.dump(lin_reg_model, 'placement_model.pkl')
print("Model saved as placement_model.pkl")

import pickle
# after training `full_pipeline`
with open('model1.pkl', 'wb') as f:
    pickle.dump(full_pipeline, f)

