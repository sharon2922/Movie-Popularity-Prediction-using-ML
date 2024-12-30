#!/usr/bin/env python
# coding: utf-8

# ### Import dataset
# 

# In[1]:


pip cache purge


# In[2]:


import pandas as pd
from pandas import read_csv
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer   
movies_clean=pd.read_csv('E:/trial and error/DATASET/CSV 5000 movies.csv')
print(movies_clean)


# In[3]:


df = movies_clean.copy()


# ### Basic Checks

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.head(5)


# In[7]:


df.tail(5)


# In[8]:


# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")


# In[9]:


df.isnull().sum()


# In[10]:


(df == 0).sum()


# In[11]:


descriptive_stats = df.describe()
print(descriptive_stats)


# In[12]:


df.describe(include=['object'])


# ### EDA
# 

# ### Univariate analysis 
# 

# ### For Numerical Datatypes:

# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of columns to plot
numerical_columns = ['Duration', 'Revenue', 'Budget', 'Vote Average', 'Vote Count']

# Create density plots
plt.figure(figsize=(14, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(df[column], shade=True)
    plt.title(f'Density Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


# In[14]:


# Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Popularity'], shade=True)
plt.title('Density Plot of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Density')
plt.show()


# ### For Categorical Datatypes

# ### Count Plot

# In[15]:


#For Language 
# Count plot for 'language'
sns.countplot(data=df, x='Language', order=df['Language'].value_counts().index)
plt.title('Count Plot of Languages')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[16]:


#FOr Genre

# Split the genres and explode into individual rows
df_genres = df['Genre'].str.split(',', expand=True).stack()
df_genres = df_genres.reset_index(drop=True)
df_genres.name = 'Genre'  # Rename the Series to 'genre'




# In[17]:


# Bar chart for 'genre'

plt.figure(figsize=(12, 5))
df_genres.value_counts().plot(kind='bar')
plt.title('Bar Chart of Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()



# ### Bivariate Analysis

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the numerical variables
numerical_vars = ['Duration', 'Revenue', 'Budget', 'Vote Average', 'Vote Count']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11.69, 8.27))  # A4 size in inches: 11.69 x 8.27
fig.suptitle('Scatter Plots of Numerical Variables vs. Popularity', fontsize=16)

# Flatten axes to iterate easily and adjust layout
axes = axes.flatten()

# Scatter plots for each numerical variable against Popularity
for i, var in enumerate(numerical_vars):
    sns.scatterplot(data=df, x=var, y='Popularity', ax=axes[i])
    axes[i].set_title(f'{var} vs. Popularity', fontsize=12)
    axes[i].set_xlabel(var, fontsize=10)
    axes[i].set_ylabel('Popularity', fontsize=10)
    axes[i].grid(True)

# Remove the last empty subplot (since there are 5 plots and 6 slots)
fig.delaxes(axes[-1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title and spacing
plt.savefig('scatter_plots_vs_popularity.png', dpi=300)  # Save as a high-resolution image
plt.show()




# In[19]:


import seaborn as sns


# ### Multivariate analysis
# 

# In[20]:


numerical_vars = ['Duration', 'Popularity', 'Revenue', 'Budget', 'Vote Average', 'Vote Count']
# Calculate correlation matrix for numerical variables
corr_matrix = df[numerical_vars].corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)  # Save with high resolution
plt.show()


# ### dropping columns 

# In[21]:


df.drop("Actors", axis=1, inplace = True)
df.drop("IMDB ID", axis=1, inplace = True)
df.drop("Overview", axis=1, inplace = True)
df.drop("Production Companies", axis=1, inplace = True)
df.drop("Country", axis=1, inplace = True)
df.drop("Tagline", axis=1, inplace = True)
df.drop("Director", axis=1, inplace = True)
df.drop("Title", axis=1, inplace = True)
df.drop('Release Date',axis=1,inplace=True)


# ### Handling missing or zero values 

# In[22]:


# List of columns to check
columns_to_check = ['Genre', 'Duration', 'Language', 
                    'Popularity', 'Revenue', 'Budget', 'Vote Average', 
                    'Vote Count']

# Dictionary to store zero counts and row indices
zero_info = {}

# Loop through each column and find rows with zero values
for column in columns_to_check:
    zero_rows = df[df[column] == 0].index.tolist()  # Get row indices where the value is zero
    zero_count = len(zero_rows)  # Count the number of zeros
    zero_info[column] = {'zero_count': zero_count, 'rows': zero_rows}

# Display the information
for column, info in zero_info.items():
    print(f"Column: {column}")
    print(f"Number of zeros: {info['zero_count']}")
    print(f"Rows with zeros: {info['rows']}\n")


# In[23]:


# Plot a histogram to see the distribution
plt.hist(df['Duration'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Movie Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()


# In[24]:


# Calculate the median of non-zero values
median_duration = df[df['Duration'] != 0]['Duration'].median()

# Replace zero values with the median
df['Duration'] = df['Duration'].replace(0, median_duration)


# ### Look at target variable 'Popularity' 

# In[25]:


df['Popularity'].describe()


# In[26]:


df.sort_values(by='Popularity',ascending=False)[:5]


# In[27]:


df.sort_values(by='Popularity',ascending=True)[:5]


# In[28]:


df.sort_values(by='Vote Average', ascending = False)[:5]


# In[29]:


df.sort_values(by='Vote Count', ascending = False)[:20]


# In[30]:


df.sort_values(by='Vote Count', ascending = True)[:20]


# In[31]:


# Drop rows where 'Vote Count' is less than 100 and update the original DataFrame
df = df[df['Vote Count'] >= 200]


# ### Feature Engineering / Creation 

# In[32]:


import seaborn as sns

sns.boxplot(df['Popularity'])
plt.xlabel('Popularity')
plt.title('Box Plot of Popularity')
plt.show()


# In[33]:


import matplotlib.pyplot as plt

plt.hist(df['Popularity'], bins=50)
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Histogram of Popularity')
plt.show()


# In[34]:


# Try different thresholds
thresholds = [10, 20, 30, 40, 50]
for threshold in thresholds:
    df['Popular'] = df['Popularity'].apply(lambda x: 1 if x > threshold else 0)
    print(f"Threshold {threshold}:")
    print(df['Popular'].value_counts())


# In[35]:


# Choose Threshold 40 or 50 based on  analysis
chosen_threshold = 40  # or 50 based on  decision

# Apply the chosen threshold to create the "Popular" column
df['Popular'] = df['Popularity'].apply(lambda x: 1 if x > chosen_threshold else 0)

# Save the updated dataset
df.to_csv('updated_dataset.csv', index=False)

# Evaluate distribution of the "Popular" column
print(df['Popular'].value_counts())


# ### Imputing outliers 

# In[36]:


# Get the names of numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Display the column names
print(numerical_columns)


# In[37]:


# Identify numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Function to identify outliers using Z-Score
def identify_outliers_zscore(df, col):
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    return df[z_scores > 3]

# Function to identify outliers using IQR
def identify_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

# Dictionary to store outliers
outliers_zscore = {}
outliers_iqr = {}

# Apply the functions to each numerical column and store the results
for col in numerical_cols:
    outliers_zscore[col] = identify_outliers_zscore(df, col)
    outliers_iqr[col] = identify_outliers_iqr(df, col)

# Print the number of outliers identified by each method for each column
for col in numerical_cols:
    print(f"Column: {col}")
    print(f"Z-Score Outliers: {len(outliers_zscore[col])}")
    print(f"IQR Outliers: {len(outliers_iqr[col])}")
    print("\n")

# Optionally, save the outliers to separate CSV files
for col in numerical_cols:
    outliers_zscore[col].to_csv(f'{col}_zscore_outliers.csv', index=False)
    outliers_iqr[col].to_csv(f'{col}_iqr_outliers.csv', index=False)


# In[38]:


# Function to impute outliers using IQR method
def impute_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Impute outliers with median
    median = df[column].median()
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median, df[column])


# Function to impute outliers using Z-Score method
def impute_outliers_zscore(df, column):
    mean = df[column].mean()
    std = df[column].std()
    
    # Calculate Z-Score
    z_scores = (df[column] - mean) / std
    
    # Impute outliers with mean
    df[column] = np.where((z_scores < -3) | (z_scores > 3), mean, df[column])

# Columns and their respective outlier counts (Z-Score and IQR)
outlier_info = {
    'Duration': {'z_score': 24, 'iqr': 54},
    'Popularity': {'z_score': 10, 'iqr': 177},
    'Revenue': {'z_score': 38, 'iqr': 142},
    'Budget': {'z_score': 32, 'iqr': 48},
    'Vote Average': {'z_score': 8, 'iqr': 10},
    'Vote Count': {'z_score': 40, 'iqr': 138}
}

# Impute outliers for each column using the best method
for column, counts in outlier_info.items():
    if counts['iqr'] > counts['z_score']:
        impute_outliers_iqr(df, column)
    else:
        impute_outliers_zscore(df, column)

# Display the first few rows of the DataFrame after imputation
print(df.head())


# ### Handling Skewness

# In[39]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# Function to calculate skewness
def calculate_skewness(column):
    return pd.Series(column).skew()  # Convert to Series if it's not

# Function to apply transformations and find the best one
def find_best_transformation(df, column_name):
    original = df[column_name]
    transformations = {
        'original': original,
        'log': np.log(original + 1),
        'sqrt': np.sqrt(original),
        'boxcox': stats.boxcox(original + 1)[0],
        'yeojohnson': PowerTransformer(method='yeo-johnson').fit_transform(df[[column_name]]).flatten()  # Flatten to 1D array
    }
    
    # Calculate skewness for each transformation
    skewness_results = {name: calculate_skewness(trans) for name, trans in transformations.items()}
    best_transformation = min(skewness_results, key=skewness_results.get)  # Find the transformation with the least skewness
    
    return transformations[best_transformation], best_transformation, skewness_results

# List of continuous numerical columns
continuous_columns = ['Duration','Popularity', 'Revenue', 'Budget', 'Vote Average', 'Vote Count']

# Apply the best transformation to each column
for column in continuous_columns:
    best_transformed, best_transformation, skewness_results = find_best_transformation(df, column)
    df[column] = best_transformed  # Update the dataframe with the best transformation
    print(f"Column: {column}")
    print("Skewness Results:")
    for name, skewness in skewness_results.items():
        print(f"  {name}: {skewness}")
    print(f"Best Transformation: {best_transformation}\n")


# ### Encoding categorical variable to numerical variable

# In[40]:


# Split genres into lists
df['Genre_onehot'] = df['Genre'].apply(lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else [])
# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# One-hot encode the genres
genre_encoded = mlb.fit_transform(df['Genre_onehot'])
genre_encoded_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)

# Join the encoded genres with the original DataFrame
df = df.join(genre_encoded_df)
print(df)


# In[41]:


df.drop('Genre',axis=1,inplace=True)
df.drop('Genre_onehot',axis=1,inplace=True)


# In[42]:


df.columns.tolist()


# In[43]:


# Assuming df is your DataFrame and 'Language' is the column name
unique_languages = df['Language'].unique()
num_unique_languages = len(unique_languages)

print(f"Number of unique languages: {num_unique_languages}")
print("Unique languages:", unique_languages)


# In[44]:


# Get a summary of unique languages and their counts
language_summary = df['Language'].value_counts().reset_index()
language_summary.columns = ['Language', 'Count']

print("Language summary:")
print(language_summary)


# In[45]:


# Create a new column for binary classification
df['Is_English'] = df['Language'].apply(lambda x: 1 if x == 'en' else 0)

# Display the updated DataFrame
print(df[['Language', 'Is_English']].head())


# In[46]:


df.drop('Language',axis=1,inplace=True)


# In[47]:


df.columns.tolist()


# ### Scaling

# In[48]:


from sklearn.preprocessing import StandardScaler

# Define the numerical columns that need scaling
numerical_features = ['Budget', 'Duration', 'Vote Count', 'Revenue','Vote Average']  # Update with your actual numerical columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical data
X_numerical = df[numerical_features]  # Replace 'df' with your DataFrame variable
X_standard_scaled = scaler.fit_transform(X_numerical)

# Add the scaled features back to the original DataFrame
df[numerical_features] = X_standard_scaled

# Display the updated DataFrame with scaled features
print("Updated DataFrame with Scaled Features:\n", df.head())


# ### Classification

# ### Logistic Regression 

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train and X_test are already scaled and available
# Split the data into features and target
X = df.drop(columns=['Popularity','Popular'])  
y = df['Popular']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)

# Predict on the training set
y_train_pred = log_reg.predict(X_train)
# Predict on the testing set
y_test_pred = log_reg.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Logistic Regression Training Accuracy:", train_accuracy)
print("Logistic Regression Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_test_pred))


# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define a more extensive parameter grid
param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)



# In[51]:


from sklearn.model_selection import cross_val_score

# Evaluate the model with cross-validation
cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')

# Print cross-validation results
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# In[52]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score

# Assuming df is your DataFrame and 'Popularity' and 'Popular' columns exist
X = df.drop(columns=['Popularity', 'Popular'])  
y = df['Popular']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Logistic Regression with the best parameters
log_reg = LogisticRegression(C=0.1, penalty='l2', solver='saga', class_weight='balanced')

# Fit the model on the training data
log_reg.fit(X_train, y_train)

# Predict on the training and testing sets
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Logistic Regression Training Accuracy:", train_accuracy)
print("Logistic Regression Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_test_pred))

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    estimator=log_reg,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate mean and standard deviation for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-Validation Score')

# Plot the fill between the curves
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

# Label the plot
plt.title('Learning Curves (Logistic Regression)')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# ### Naive Bayes

# In[53]:


from sklearn.naive_bayes import GaussianNB


# Initialize and train Gaussian Naive Bayes model
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Predict on the training set
y_train_pred = naive_bayes.predict(X_train)
# Predict on the testing set
y_test_pred = naive_bayes.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Naive Bayes Training Accuracy:", train_accuracy)
print("Naive Bayes Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_test_pred))


# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Assuming df is your DataFrame and has been loaded
# Split the data into features and target
X = df.drop(columns=['Popularity', 'Popular'])
y = df['Popular']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Initialize Gaussian Naive Bayes model
naive_bayes = GaussianNB()

# Create a pipeline that first applies SMOTE and then fits the model
pipeline = Pipeline([
    ('smote', smote),
    ('classifier', naive_bayes)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the training set
y_train_pred = pipeline.predict(X_train)
# Predict on the testing set
y_test_pred = pipeline.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Naive Bayes Training Accuracy with SMOTE:", train_accuracy)
print("Naive Bayes Testing Accuracy with SMOTE:", test_accuracy)
print("\nClassification Report for Testing Set with SMOTE:")
print(classification_report(y_test, y_test_pred))

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation for plotting
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curves
plt.figure()
plt.title("Learning Curve (Naive Bayes with SMOTE)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# ### Random Forest

# In[55]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predict on the training set
y_train_pred = random_forest.predict(X_train)
# Predict on the testing set
y_test_pred = random_forest.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Random Forest Training Accuracy:", train_accuracy)
print("Random Forest Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_test_pred))


# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming df is your DataFrame and has been loaded
# Split the data into features and target
X = df.drop(columns=['Popularity', 'Popular'])
y = df['Popular']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=random_forest, 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=2, 
                           scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Retrieve the best model and parameters
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict on the training and testing sets using the best model
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
print("Best Hyperparameters:", best_params)
print("Random Forest Training Accuracy:", train_accuracy)
print("Random Forest Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_test_pred))

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    best_rf_model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation for plotting
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curves
plt.figure()
plt.title("Learning Curve (Random Forest Classifier)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# ### Regression 

# ### features for linear , lasso and ridge regression

# In[57]:


from sklearn.feature_selection import SelectKBest, f_regression
X = df.drop(columns=['Popularity','Popular'])
y = df['Popularity']
# Initialize SelectKBest with the number of features to select
selector = SelectKBest(f_regression, k=10)

# Fit the selector to the data
selector.fit(X, y)

# Get boolean mask of selected features
mask = selector.get_support()

# Get the feature names
feature_names = X.columns  # Assuming X is a DataFrame

# Print the names of the selected features
selected_features = feature_names[mask]
print("Top 10 selected features:")
print(selected_features)


# In[ ]:





# ### Linear regression

# In[58]:


features_df = df[['Duration', 'Budget', 'Vote Average', 'Vote Count', 'Action',
       'Adventure', 'Animation', 'Crime', 'Drama', 'Is_English']]
labels_df = df['Popularity']


# In[59]:


labels_df = labels_df[features_df.index]


# In[60]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming features_df and labels_df are your feature and target DataFrames
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
model = RandomForestRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = model.predict(X_train)

# Predict on the testing set
y_test_pred = model.predict(X_test)

# Calculate scores for training data
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

# Calculate scores for testing data
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Print the scores
print(f"Training R2 Score: {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing R2 Score: {test_r2:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")


# In[61]:


# Import necessary libraries for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter grid for Randomized Search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Setup RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings to sample
    cv=3,       # 3-fold cross-validation
    verbose=2,  # Higher verbosity to display progress
    random_state=42,
    n_jobs=-1   # Use all available cores
)

# Fit the RandomizedSearchCV to the data
rf_random.fit(X_train, y_train)

# Best parameters and best score from RandomizedSearchCV
print(f"Best Parameters: {rf_random.best_params_}")
print(f"Best Score: {rf_random.best_score_:.4f}")

# Predict using the best model
best_model = rf_random.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate and print the scores for the best model
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print(f"Training R2 Score: {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing R2 Score: {test_r2:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")


# In[62]:


# Import necessary libraries
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = model.predict(X_train)

# Predict on the testing set
y_test_pred = model.predict(X_test)

# Calculate scores for training data
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

# Calculate scores for testing data
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Print the scores
print(f"Training R2 Score: {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing R2 Score: {test_r2:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation for the learning curve
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure()
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

# Plot the fill between lines
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve for Linear Regression')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# ### RFE Recursive Feature Elimination.

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve

# Initialize the Linear Regression model
model = LinearRegression()

# Initialize RFE with the model and specify the number of features to select
# Adjust n_features_to_select based on the number of features you want
rfe = RFE(estimator=model, n_features_to_select=5)  # Adjust this number as needed

# Fit RFE on the training data
rfe.fit(X_train, y_train)

# Get the boolean mask of selected features
selected_features = rfe.support_

# Get the feature names for selected features
selected_feature_names = X_train.columns[selected_features]

print("Selected Features:", selected_feature_names)

# Filter the training and test data to include only the selected features
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

# Refit the model on the selected features
model.fit(X_train_selected, y_train)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# Calculate and print scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Training R2 Score with RFE: {train_r2:.4f}")
print(f"Testing R2 Score with RFE: {test_r2:.4f}")

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_selected, y_train, cv=5, scoring='r2', 
    train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Calculate the mean and standard deviation for training and testing scores
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plot the learning curves
plt.figure()
plt.title("Learning Curves (Linear Regression)")
plt.xlabel("Training examples")
plt.ylabel("R2 Score")
plt.ylim(0, 1.1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# ### without features 

# In[64]:


X = df.drop(columns=['Popularity','Popular'])  
y = df['Popularity']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=42)


# ### XGBoost

# In[65]:


from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Initialize the XGBoost Regressor with default parameters
xgb_model = XGBRegressor()

# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Calculate and print R2 scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Training R2 Score with XGBoost Regressor: {train_r2:.4f}")
print(f"Testing R2 Score with XGBoost Regressor: {test_r2:.4f}")


# ### Hyperparameter tuning for XGBoost

# In[66]:


import numpy as np 
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error

# Define the model
xgb_model = XGBRegressor()

# Define the hyperparameters and their possible values to tune
param_grid = {
    'n_estimators': [100, 200, 300],       # Number of boosting rounds
    'max_depth': [3, 5, 7],                # Maximum depth of a tree
    'learning_rate': [0.01, 0.1, 0.2],     # Learning rate (step size shrinkage)
    'subsample': [0.6, 0.8, 1.0],          # Fraction of samples used for fitting individual trees
    'colsample_bytree': [0.6, 0.8, 1.0],   # Fraction of features used in each boosting round
    'gamma': [0, 0.1, 0.2]                 # Minimum loss reduction required to make a split
}

# Initialize GridSearchCV with the XGBoost model and the defined parameter grid
grid_search = GridSearchCV(estimator=xgb_model, 
                           param_grid=param_grid, 
                           scoring='r2', 
                           cv=5,             # 5-fold cross-validation
                           n_jobs=-1,        # Use all available cores
                           verbose=2)

# Fit the model on the training data with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Retrieve the best model and parameters from the grid search
best_xgb_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict using the best model
y_train_pred = best_xgb_model.predict(X_train)
y_test_pred = best_xgb_model.predict(X_test)

# Calculate and print the R2 scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate RMSE for training and testing sets
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Best Hyperparameters: {best_params}")
print(f"Training R2 Score with Tuned XGBoost Regressor: {train_r2:.4f}")
print(f"Testing R2 Score with Tuned XGBoost Regressor: {test_r2:.4f}")
print(f"Training RMSE with Tuned XGBoost Regressor: {train_rmse:.4f}")
print(f"Testing RMSE with Tuned XGBoost Regressor: {test_rmse:.4f}")

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    best_xgb_model, X, y, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation for plotting
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curves
plt.figure()
plt.title("Learning Curve (XGBoost Regressor)")
plt.xlabel("Training examples")
plt.ylabel("R2 Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# ### Adaboost

# In[67]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score

# Initialize the AdaBoost Regressor with default parameters
ada_model = AdaBoostRegressor()

# Fit the model on the training data
ada_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = ada_model.predict(X_train)
y_test_pred = ada_model.predict(X_test)

# Calculate and print R² scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training R2 Score with AdaBoost Regressor: {train_r2:.4f}")
print(f"Testing R2 Score with AdaBoost Regressor: {test_r2:.4f}")


# ### Hyperparameter Tuning for Adaboost

# In[68]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the model
ada_model = AdaBoostRegressor()

# Define the hyperparameters and their possible values to tune
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of boosting stages
    'learning_rate': [0.01, 0.1, 1.0],        # Learning rate
}

# Initialize GridSearchCV with the AdaBoost model and the defined parameter grid
grid_search = GridSearchCV(estimator=ada_model, 
                           param_grid=param_grid, 
                           scoring='r2', 
                           cv=5,             # 5-fold cross-validation
                           n_jobs=-1,        # Use all available cores
                           verbose=2)

# Fit the model on the training data with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Retrieve the best model and parameters from the grid search
best_ada_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict using the best model
y_train_pred = best_ada_model.predict(X_train)
y_test_pred = best_ada_model.predict(X_test)

# Calculate and print the R² scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Compute RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

train_rmse = compute_rmse(y_train, y_train_pred)
test_rmse = compute_rmse(y_test, y_test_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Training R² Score with Tuned AdaBoost Regressor: {train_r2:.4f}")
print(f"Testing R² Score with Tuned AdaBoost Regressor: {test_r2:.4f}")
print(f"Training RMSE with Tuned AdaBoost Regressor: {train_rmse:.4f}")
print(f"Testing RMSE with Tuned AdaBoost Regressor: {test_rmse:.4f}")

# Learning Curve for AdaBoost Regressor
train_sizes, train_scores, test_scores = learning_curve(
    best_ada_model, X_train, y_train, cv=5, scoring='r2', 
    train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_std = test_scores.std(axis=1)

# Compute RMSE for different training sizes
train_rmse_list = []
test_rmse_list = []
for size in train_sizes:
    # Fit model on a subset of training data
    X_subset = X_train[:int(size * len(X_train))]
    y_subset = y_train[:int(size * len(y_train))]
    best_ada_model.fit(X_subset, y_subset)
    y_train_pred_subset = best_ada_model.predict(X_subset)
    y_test_pred_subset = best_ada_model.predict(X_test)
    
    train_rmse_list.append(compute_rmse(y_subset, y_train_pred_subset))
    test_rmse_list.append(compute_rmse(y_test, y_test_pred_subset))

# Plotting the learning curves for R² and RMSE
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot R² Learning Curves
ax1.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training R² Score')
ax1.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation R² Score')
ax1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
ax1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
ax1.set_xlabel('Training Size')
ax1.set_ylabel('R² Score')
ax1.set_title('Learning Curves (AdaBoost Regressor)')
ax1.legend(loc='upper left')
ax1.grid()

# Create a second y-axis for RMSE
ax2 = ax1.twinx()
ax2.plot(train_sizes, train_rmse_list, 'o--', color='b', label='Training RMSE')
ax2.plot(train_sizes, test_rmse_list, 'o--', color='orange', label='Testing RMSE')
ax2.set_ylabel('RMSE')
ax2.legend(loc='upper right')

plt.show()


# ### Decision Tree Regressor

# In[69]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Initialize the Decision Tree Regressor
dt_model = DecisionTreeRegressor()

# Fit the model on the training data
dt_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# Calculate and print R² scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training R2 Score with Decision Tree Regressor: {train_r2:.4f}")
print(f"Testing R2 Score with Decision Tree Regressor: {test_r2:.4f}")


# In[70]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the Decision Tree Regressor
dt_model = DecisionTreeRegressor()

# Define the hyperparameters and their possible values to tune
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required to be at a leaf node
    'max_features': [None, 'sqrt', 'log2'],   # Number of features to consider when looking for the best split
    'criterion': ['mse', 'friedman_mse', 'mae']  # Function to measure the quality of a split
}

# Initialize GridSearchCV with the Decision Tree model and the defined parameter grid
grid_search = GridSearchCV(estimator=dt_model, 
                           param_grid=param_grid, 
                           scoring='r2', 
                           cv=5,             # 5-fold cross-validation
                           n_jobs=-1,        # Use all available cores
                           verbose=2)

# Fit the model on the training data with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Retrieve the best model and parameters from the grid search
best_dt_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict using the best model
y_train_pred = best_dt_model.predict(X_train)
y_test_pred = best_dt_model.predict(X_test)

# Calculate and print the R² scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Compute RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

train_rmse = compute_rmse(y_train, y_train_pred)
test_rmse = compute_rmse(y_test, y_test_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Training R² Score with Tuned Decision Tree Regressor: {train_r2:.4f}")
print(f"Testing R² Score with Tuned Decision Tree Regressor: {test_r2:.4f}")
print(f"Training RMSE with Tuned Decision Tree Regressor: {train_rmse:.4f}")
print(f"Testing RMSE with Tuned Decision Tree Regressor: {test_rmse:.4f}")

# Plot Learning Curves
train_sizes, train_scores, test_scores = learning_curve(best_dt_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

# Calculate mean and std of training and test scores
train_mean_r2 = train_scores.mean(axis=1)
train_std_r2 = train_scores.std(axis=1)
test_mean_r2 = test_scores.mean(axis=1)
test_std_r2 = test_scores.std(axis=1)

# Compute RMSE for different training sizes
train_rmse_list = []
test_rmse_list = []
for size in train_sizes:
    # Fit model on a subset of training data
    X_subset = X_train[:int(size * len(X_train))]
    y_subset = y_train[:int(size * len(y_train))]
    best_dt_model.fit(X_subset, y_subset)
    y_train_pred_subset = best_dt_model.predict(X_subset)
    y_test_pred_subset = best_dt_model.predict(X_test)
    
    train_rmse_list.append(compute_rmse(y_subset, y_train_pred_subset))
    test_rmse_list.append(compute_rmse(y_test, y_test_pred_subset))




# In[71]:


# Plotting the learning curves for R² and RMSE
fig, ax1 = plt.subplots(figsize=(8, 8))

# Plot R² Learning Curves
ax1.plot(train_sizes, train_mean_r2, 'o-', color='r', label='Training R² Score')
ax1.plot(train_sizes, test_mean_r2, 'o-', color='g', label='Cross-validation R² Score')
ax1.fill_between(train_sizes, train_mean_r2 - train_std_r2, train_mean_r2 + train_std_r2, alpha=0.1, color='r')
ax1.fill_between(train_sizes, test_mean_r2 - test_std_r2, test_mean_r2 + test_std_r2, alpha=0.1, color='g')
ax1.set_xlabel('Training Size')
ax1.set_ylabel('R² Score')
ax1.set_title('Learning Curves (Decision Tree Regressor)')
ax1.legend(loc='upper left')
ax1.grid()

# Create a second y-axis for RMSE
ax2 = ax1.twinx()
ax2.plot(train_sizes, train_rmse_list, 'o--', color='b', label='Training RMSE')
ax2.plot(train_sizes, test_rmse_list, 'o--', color='orange', label='Testing RMSE')
ax2.set_ylabel('RMSE')
ax2.legend(loc='upper right')

plt.show()


# ### SVR

# In[72]:


import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
import numpy as np

# Initialize the SVR model with default parameters
svr_model = SVR()

# Fit the model on the training data
svr_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = svr_model.predict(X_train)
y_test_pred = svr_model.predict(X_test)

# Calculate and print R2 scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Compute RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

train_rmse = compute_rmse(y_train, y_train_pred)
test_rmse = compute_rmse(y_test, y_test_pred)

print(f"Training R² Score with SVR: {train_r2:.4f}")
print(f"Testing R² Score with SVR: {test_r2:.4f}")
print(f"Training RMSE with SVR: {train_rmse:.4f}")
print(f"Testing RMSE with SVR: {test_rmse:.4f}")

# Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    svr_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and std of training and test scores
train_mean_r2 = train_scores.mean(axis=1)
train_std_r2 = train_scores.std(axis=1)
test_mean_r2 = test_scores.mean(axis=1)
test_std_r2 = test_scores.std(axis=1)

# Compute RMSE for different training sizes
train_rmse_list = []
test_rmse_list = []
for size in train_sizes:
    # Fit model on a subset of training data
    X_subset = X_train[:int(size * len(X_train))]
    y_subset = y_train[:int(size * len(y_train))]
    svr_model.fit(X_subset, y_subset)
    y_train_pred_subset = svr_model.predict(X_subset)
    y_test_pred_subset = svr_model.predict(X_test)
    
    train_rmse_list.append(compute_rmse(y_subset, y_train_pred_subset))
    test_rmse_list.append(compute_rmse(y_test, y_test_pred_subset))

# Plotting the learning curves for R² and RMSE
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot R² Learning Curves
ax1.plot(train_sizes, train_mean_r2, 'o-', color='r', label='Training R² Score')
ax1.plot(train_sizes, test_mean_r2, 'o-', color='g', label='Cross-validation R² Score')
ax1.fill_between(train_sizes, train_mean_r2 - train_std_r2, train_mean_r2 + train_std_r2, alpha=0.1, color='r')
ax1.fill_between(train_sizes, test_mean_r2 - test_std_r2, test_mean_r2 + test_std_r2, alpha=0.1, color='g')
ax1.set_xlabel('Training Size')
ax1.set_ylabel('R² Score')
ax1.set_title('Learning Curves (Support Vector Regressor)')
ax1.legend(loc='upper left')
ax1.grid()

# Create a second y-axis for RMSE
ax2 = ax1.twinx()
ax2.plot(train_sizes, train_rmse_list, 'o--', color='b', label='Training RMSE')
ax2.plot(train_sizes, test_rmse_list, 'o--', color='orange', label='Testing RMSE')
ax2.set_ylabel('RMSE')
ax2.legend(loc='upper right')

plt.show()


# In[73]:


import pandas as pd

# Classification Results
classification_results = {
    'Model': ['Logistic Regression', 'Naive Bayes (SMOTE)', 'Random Forest'],
    'Training Accuracy': [0.7541, 0.3224, 0.9771],
    'Testing Accuracy': [0.7590, 0.3162, 0.8524],
    'Precision (Class 0)': [0.93, 0.84, 0.87],
    'Precision (Class 1)': [0.50, 0.25, 0.77],
    'Recall (Class 0)': [0.74, 0.13, 0.95],
    'Recall (Class 1)': [0.82, 0.93, 0.54],
    'F1-Score (Class 0)': [0.82, 0.22, 0.91],
    'F1-Score (Class 1)': [0.62, 0.39, 0.64],
    'Accuracy': [0.76, 0.32, 0.85],
    
}

# Create DataFrame
classification_df = pd.DataFrame(classification_results)

# Regression Results
regression_results = {
    'Model': ['Multilinear Regression', 'XGBoost', 'AdaBoost', 'Decision Tree Regressor', 'SVR'],
    'Training R² Score': [0.2799, 0.4547, 0.3279, 0.4468, 0.0610],
    'Testing R² Score': [0.2927, 0.3554, 0.3332, 0.2297, 0.0638],
    'Training RMSE': [0.0391, 'N/A', 'N/A', 'N/A', 'N/A'],
    'Testing RMSE': [0.0380, 'N/A', 'N/A', 'N/A', 'N/A']
}

# Create DataFrame
regression_df = pd.DataFrame(regression_results)

# Display the Classification Results Table
print("Classification Models Performance:")
print(classification_df.to_markdown(index=False))

# Display the Regression Results Table
print("\nRegression Models Performance:")
print(regression_df.to_markdown(index=False))


# In[74]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Classification Results Data
classification_data = {
    'Model': ['LR', 'NB(SMOTE)', 'RF'],
    'Training Accuracy': [0.7541, 0.3224, 0.9771],
    'Testing Accuracy': [0.7590, 0.3162, 0.8524],
    'Precision (0)': [0.93, 0.84, 0.87],
    'Precision (1)': [0.50, 0.25, 0.77],
    'Recall (0)': [0.74, 0.13, 0.95],
    'Recall (1)': [0.82, 0.93, 0.54],
    'F1-Score (0)': [0.82, 0.22, 0.91],
    'F1-Score (1)': [0.62, 0.39, 0.64],
    'Accuracy': [0.76, 0.32, 0.85],
    
}

# Updated Regression Results Data
regression_data = {
    'Model': ['Multilinear Regression', 'XGBoost', 'AdaBoost', 'Decision Tree Regressor', 'SVR'],
    'Training R² Score': [0.2799, 0.4547, 0.3267, 0.4165, 0.0610],
    'Testing R² Score': [0.2927, 0.3554, 0.3296, 0.2040, 0.0638],
    'Training RMSE': [0.0391, 0.0340, 0.0378, 0.0352, 0.0446],  # Updated RMSE values
    'Testing RMSE': [0.0380, 0.0363, 0.0370, 0.0403, 0.0437]   # Updated RMSE values
}

# Convert to DataFrames
classification_df = pd.DataFrame(classification_data)
regression_df = pd.DataFrame(regression_data)

# Plotting Classification Table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
table = ax.table(cellText=classification_df.values,
                 colLabels=classification_df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Classification Models Performance')
plt.show()

# Plotting Regression Table
fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted size for readability
ax.axis('off')
table = ax.table(cellText=regression_df.values,
                 colLabels=regression_df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Regression Models Performance')
plt.show()


# In[75]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Classification Results Data
classification_data = {
    'Model': ['LR', 'NB(SMOTE)', 'RF'],
    'Training Accuracy': [0.7541, 0.3224, 0.9771],
    'Testing Accuracy': [0.7590, 0.3162, 0.8524],
    'Precision (0)': [0.93, 0.84, 0.87],
    'Precision (1)': [0.50, 0.25, 0.77],
    'Recall (0)': [0.74, 0.13, 0.95],
    'Recall (1)': [0.82, 0.93, 0.54],
    'F1-Score (0)': [0.82, 0.22, 0.91],
    'F1-Score (1)': [0.62, 0.39, 0.64],
    'Accuracy': [0.76, 0.32, 0.85],
}

# Convert to DataFrame
classification_df = pd.DataFrame(classification_data)

# Plotting Classification Table
fig, ax = plt.subplots(figsize=(16, 8))  # Increase the width for better fit
ax.axis('off')
table = ax.table(cellText=classification_df.values,
                 colLabels=classification_df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

# Auto-size font and scale for readability
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 1.3)  # Increase scaling to fit content better

# Adding padding to the cell text to enhance spacing
for key, cell in table._cells.items():
    cell.set_text_props(verticalalignment='center', horizontalalignment='center', fontsize=10)
    cell.set_edgecolor('black')

plt.title('Classification Models Performance')
plt.show()

