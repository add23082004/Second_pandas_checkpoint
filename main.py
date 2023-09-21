import pandas as pd
from sklearn.preprocessing import LabelEncoder

space = ('\n  ')

# Load the dataset, display the ten first lines, store the results in a variable called 'client_0_bills'.
client_0_bills = pd.read_csv('/Users/amadoudiakhadiop/Documents/STBH.csv', low_memory=False)
print(client_0_bills.head(10), space)

# What is the data type of the 'client_0_bills' variable ?
print(type(client_0_bills), "is the data type of the variable.", space)

# Display the general information of the dataset and try to answer the following questions
print(client_0_bills.info(), space)

# How many rows and columns do we have in this dataset ?
print("The number of rows and columns are respectively: ", client_0_bills.shape, space)

# How many categorical features are present in the dataset ?
categorical_feature_count = 0
for column in client_0_bills.columns:
    if client_0_bills[column].dtype == 'object':
        categorical_feature_count += 1
print("Number of categorical features:", categorical_feature_count, space)

# How much memory space does the dataset consume?
memory_usage_MB = client_0_bills.memory_usage().sum() / (1024 ** 2)
print("The dataset consumes", memory_usage_MB, "MB", space)

# Inspect the dataset for potential missing values.
missing_values = client_0_bills.isna().sum()

# Display columns with missing values, along with the count of missing values in each column
print("Columns with missing values:")
print(missing_values[missing_values > 0], space)

# As these missing values are numerical, we will handle the situation by replacing
# the missing values with the mean of the columns.
for column in client_0_bills.columns:
    if client_0_bills[column].isna().any():
        column_mean = client_0_bills[column].mean()
        client_0_bills[column].fillna(column_mean, inplace=True)

# Run a descriptive analysis on numeric features (columns).
numeric_columns = client_0_bills.select_dtypes(include=['number'])
descriptive_stats = numeric_columns.describe()
print(descriptive_stats)

# Select the bills records for the client with an id ='train_Client_O', using 2 methods.
# Using the query() method to filter records
method_1_result = client_0_bills.query("client_id == 'train_Client_0'")
print(method_1_result, space)
# Using .loc[] to filter records
method_2_result = client_0_bills.loc[client_0_bills['client_id'] == 'train_Client_0']
print(method_2_result, space)

# Transform the 'counter_type' feature to a numeric variable using the encoder of your choice.
# Using scikit-learn
client_0_bills['counter_type_encoded'] = LabelEncoder().fit_transform(client_0_bills['counter_type'])
print(client_0_bills, space)
# Using pandas
# client_0_bills = pd.get_dummies(client_0_bills, columns=['counter_type'])

# Delete the 'counter statue' feature from the Dataframe
client_0_bills.drop(columns=['counter_statue'], inplace=True)
print("counter_statue table dropped ")
