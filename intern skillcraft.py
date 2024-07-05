#!/usr/bin/env python
# coding: utf-8

# # Q.1 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Load the new dataset
file_path_new = "C:\\Users\\asd\\Downloads\\HRDataset_v14.csv"
data_new = pd.read_csv(file_path_new)

# Display the first few rows of the new dataset and general information
data_new_info = data_new.info()
data_new_head = data_new.head()

data_new_info, data_new_head


# In[40]:


import pandas as pd

# Loading the dataset
file_path_new = "C:\\Users\\asd\\Downloads\\HRDataset_v14.csv"
data_new = pd.read_csv(file_path_new)

# Displaying the first few rows of the dataset and general information
print(data_new.info())
print(data_new.head())



# ## gender distribution
# 
# 
# 

# In[15]:


import matplotlib.pyplot as plt

# Bar graph for gender distribution
gender_counts = data_new['GenderID'].value_counts()
gender_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['1 = male, 0= female'])
plt.show()


# # Q.2

# In[17]:


import pandas as pd

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_df = pd.read_csv(url)

# Display the rows of the dataset
titanic_df


# In[18]:


# Check for missing values in each column
print(titanic_df.isnull().sum())


# ## cleaning data

# In[22]:


# deleting cabin column
titanic_df.drop(columns=["Cabin"], inplace=True)


# In[25]:


# filling mode in embraked 
titanic_df["Embarked"].fillna(titanic_df["Embarked"].mode()[0], inplace=True)
# filling median in age 
titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)



# In[27]:


# Histogram of age distribution
titanic_df["Age"].plot(kind="hist", bins=20)


# In[33]:


# Scatter plot: Age vs. Fare
titanic_df.plot.scatter(x="Age", y="Fare",c='red')


# In[39]:


# urvival Rate by Gender
import seaborn as sns
sns.barplot(x="Sex", y="Survived", data=titanic_df)



# In[40]:


#Survival Rate by Passenger Class (Pclass):
sns.barplot(x="Pclass", y="Survived", data=titanic_df)


# In[45]:


#Fare vs. Survival:
sns.barplot(y="Fare", x="Survived", data=titanic_df)


# In[47]:


sns.countplot(x="Embarked", hue="Survived", data=titanic_df)


# ## Q.3

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load the dataset
path='C:\\Users\\asd\\Downloads\\bank.csv'
data = pd.read_csv(path)

# Preprocess the data
X = data.drop('deposit', axis=1)
y = data['deposit']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[2]:


plt.figure(figsize=(8, 8))
plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()


# In[3]:


# Create the classifier  
clf = DecisionTreeClassifier(ccp_alpha=0.01)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[4]:


plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()


# ## Q.4

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the CSV file
file_path = "C:\\Users\\asd\\Downloads\\accidents.csv"
accident_data = pd.read_csv(file_path, parse_dates=["AccidentDate"])

# Display basic information about the dataset
print(accident_data.info())

# Display summary statistics for numerical columns
print(accident_data.describe())
#checking first five rows
print(accident_data.head())


# In[23]:


plt.figure(figsize=(11, 6))
sns.countplot(x="State", data=accident_data, order=accident_data['State'].value_counts().index)
plt.title("Distribution of Accidents Across States")
plt.xticks(rotation=50, ha="right")
plt.show()


# In[25]:


# Explore the distribution of accidents based on weather conditions
plt.figure(figsize=(8, 6))
sns.countplot(x="WeatherCondition", data=accident_data, order=accident_data['WeatherCondition'].value_counts().index)
plt.title("Distribution of Accidents Based on Weather Conditions")
plt.show()


# In[26]:


# Explore the distribution of accidents based on road conditions
plt.figure(figsize=(8, 6))
sns.countplot(x="RoadCondition", data=accident_data, order=accident_data['RoadCondition'].value_counts().index)
plt.title("Distribution of Accidents Based on Road Conditions")
plt.show()


# In[27]:


# Explore the distribution of accidents based on the timing of the day
plt.figure(figsize=(8, 6))
sns.countplot(x="Timing", data=accident_data, order=accident_data['Timing'].value_counts().index)
plt.title("Distribution of Accidents Based on Timing of the Day")
plt.show()


# In[29]:


# Explore the reasons for accidents
plt.figure(figsize=(12, 6))
sns.countplot(x="Reason", data=accident_data, order=accident_data['Reason'].value_counts().index)
plt.title("Distribution of Accidents Based on Reasons")
plt.xticks(rotation=40, ha="right")
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Explore patterns related to road conditions
plt.figure(figsize=(12, 6))
sns.countplot(x="RoadCondition", hue="Timing", data=accident_data, order=accident_data['RoadCondition'].value_counts().index)
plt.title("Accidents by Road Conditions and Timing of the Day")
plt.xticks(rotation=45, ha="right")
plt.show()


# In[31]:


# Explore patterns related to weather conditions
plt.figure(figsize=(12, 6))
sns.countplot(x="WeatherCondition", hue="Timing", data=accident_data, order=accident_data['WeatherCondition'].value_counts().index)
plt.title("Accidents by Weather Conditions and Timing of the Day")
plt.show()


# In[32]:


# Explore patterns related to time of day
plt.figure(figsize=(10, 6))
sns.countplot(x="Timing", data=accident_data, order=accident_data['Timing'].value_counts().index)
plt.title("Accidents by Timing of the Day")
plt.show()


# In[33]:


# Set the style for seaborn plots
sns.set(style="whitegrid")

# Visualize accident hotspots (States with higher accident frequencies)
plt.figure(figsize=(14, 8))
state_accidents = accident_data['State'].value_counts()
sns.barplot(x=state_accidents.index, y=state_accidents.values, palette="viridis")
plt.title("Accident Hotspots by State")
plt.xlabel("State")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.show()


# In[34]:


# Visualize contributing factors (Reasons for accidents)
plt.figure(figsize=(12, 6))
reasons_accidents = accident_data['Reason'].value_counts()
sns.barplot(x=reasons_accidents.index, y=reasons_accidents.values, palette="muted")
plt.title("Contributing Factors for Accidents")
plt.xlabel("Reason")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.show()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Group the data by state and timing, and calculate the total number of deaths
state_timing_deaths = accident_data.groupby(['State', 'Timing'])['Deaths'].sum().reset_index()

# Pivot the table to get a format suitable for plotting
state_timing_deaths_pivot = state_timing_deaths.pivot(index='State', columns='Timing', values='Deaths').fillna(0)

# Create a bar plot for state-wise total death timing ratio
plt.figure(figsize=(40, 8))
state_timing_deaths_pivot.plot(kind='bar', stacked=True, colormap="viridis")
plt.title("Total Death Timing Ratio")
plt.xlabel("States")
plt.ylabel("Total Deaths")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Timing")
plt.show()


# In[ ]:




