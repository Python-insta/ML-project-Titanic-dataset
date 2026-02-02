# Required Modules.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Need to add more model and compare its accuracy.

# # Data preprocessing
titanic_data = pd.read_csv('F:/titanic.csv')  # Load The dataset.
titanic_data['Male'] = list(map(int, titanic_data['Sex'] == 'male'))
# Change the Category feature into numerical.
titanic_data = titanic_data.drop('Sex', axis=1)

print(titanic_data.head(10))  # Check top 10 lines in dataset.

titanic_data.to_csv('F:/Titanic_processed.csv', index=False)
# Save the dataset.


# # Train-Test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in split.split(titanic_data, titanic_data['Pclass']):
    # Splits the sample on the base of 'Pclass'.
    strat_train = titanic_data.loc[train_index]
    strat_test = titanic_data.loc[test_index]
    strat_train.to_csv('F:/Training_data.csv', index=False)
    # Save Training data in separate csv.
    strat_test.to_csv('F:/Testing_data.csv', index=False)
    # Save Testing data in separate csv.
del titanic_data

# # Data Visualization
train_data = pd.read_csv('F:/Training_data.csv')
train_data.hist()  # Plot histogram of all features.
plt.show()
for attr in ['Pclass', 'Male', 'Fare']:
    plt.scatter(train_data[attr], train_data['Age'], alpha=0.05,
                c=train_data['Survived'], cmap=plt.get_cmap('jet'))
    # Plot scatter plot with respect to 'Age'.
    plt.xlabel(attr)
    plt.ylabel('Age')
    plt.show()

# # Looking for Correlation in features
corr_matrix = train_data.corr()
print(corr_matrix['Survived'].sort_values(ascending=False))

# # Training and Validation
train_data = pd.read_csv('F:/Training_data.csv')
y_label = train_data['Survived']
# Extract target labels.
train_data = train_data.drop('Survived', axis=1)
# Remove target from training data.
system = RandomForestClassifier()
validation_score = cross_val_score(system, train_data, y_label, scoring='accuracy', cv=10)
print(sum(validation_score)/10)

# # Testing
y_label = train_data['Survived']
# Extract target labels.
train_data = train_data.drop('Survived', axis=1)
# Remove target from training data.
test_data = pd.read_csv('F:/Testing_data.csv')
y_test = test_data['Survived']
# Extract target labels.
test_data = test_data.drop('Survived', axis=1)
# Remove target from testing data.

system = RandomForestClassifier()
system.fit(train_data, y_label)
result = system.predict(test_data)
print('Model Performance:', accuracy_score(y_test, result))
