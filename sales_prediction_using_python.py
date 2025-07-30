
"""# Task
Analyze the provided advertising dataset located at "/content/Advertising.csv", visualize the relationships between advertising spend on different platforms and sales, build and train a linear regression model to predict sales based on advertising spend, evaluate the model's performance, and use the trained model to predict sales.

## Load the data

### Subtask:
Load the provided CSV file into a pandas DataFrame.

**Reasoning**:
Import the pandas library and load the data into a pandas DataFrame.
"""

import pandas as pd

df = pd.read_csv('/content/Advertising.csv')

"""**Reasoning**:
Display the first few rows of the DataFrame to verify the data loading.


"""

df.head()

"""## Analyze the data

### Subtask:
Display the first 5 rows, check for missing values, and display descriptive statistics of the dataset.

**Reasoning**:
Display the first 5 rows, check for missing values, and display descriptive statistics of the dataset.
"""

display(df.head())
display(df.isnull().sum())
display(df.describe())

"""## Visualize the data

### Subtask:
Create a scatter plot to visualize the relationship between TV, radio, and newspaper advertising spend and sales.

**Reasoning**:
Import the necessary libraries for plotting and create scatter plots to visualize the relationship between advertising spend on different platforms and sales.
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x='TV', y='Sales', data=df)
plt.title('TV Advertising Spend vs Sales')
plt.show()

sns.scatterplot(x='Radio', y='Sales', data=df)
plt.title('Radio Advertising Spend vs Sales')
plt.show()

sns.scatterplot(x='Newspaper', y='Sales', data=df)
plt.title('Newspaper Advertising Spend vs Sales')
plt.show()

"""## Prepare the data

### Subtask:
Split the data into training and testing sets.

**Reasoning**:
Define features and target, import train_test_split, and split the data into training and testing sets.
"""

from sklearn.model_selection import train_test_split

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Build and train the model

### Subtask:
Use a linear regression model to predict sales based on advertising spend.

**Reasoning**:
Import the LinearRegression class, instantiate a model, and fit it to the training data.
"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

"""## Evaluate the model

### Subtask:
Evaluate the model's performance using appropriate metrics.

**Reasoning**:
Calculate and print the Mean Squared Error and R-squared score to evaluate the model's performance on the test set.
"""

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2:.2f}')

"""## Predict on new data

### Subtask:
Use the trained model to make predictions on new data.

**Reasoning**:
Create a new DataFrame with sample advertising spend values for prediction and use the trained model to predict sales for this new data.
"""

import pandas as pd

new_data = pd.DataFrame({
    'TV': [100, 200, 50],
    'Radio': [20, 40, 10],
    'Newspaper': [10, 20, 5]
})

predicted_sales = model.predict(new_data)

print("Predicted Sales:")
print(predicted_sales)

"""## Summary:

### Data Analysis Key Findings

*   The dataset contains 200 entries with no missing values.
*   There appears to be a positive linear relationship between TV advertising spend and sales, and Radio advertising spend and sales, as indicated by the scatter plots.
*   The relationship between Newspaper advertising spend and sales appears weaker or less linear compared to TV and Radio.
*   The linear regression model achieved a Mean Squared Error (MSE) of 3.17 on the test set, indicating the average squared difference between predicted and actual sales is relatively low.
*   The R-squared score of 0.90 suggests that approximately 90% of the variance in sales can be explained by the advertising spend on TV, Radio, and Newspaper.
*   The trained model successfully predicted sales values for new, unseen advertising spend data.

### Insights or Next Steps

*   TV and Radio advertising appear to be strong predictors of sales, while Newspaper advertising has a less clear impact.
*   Further analysis could investigate potential interactions between advertising platforms or explore non-linear relationships to potentially improve the model's predictive power.

"""
