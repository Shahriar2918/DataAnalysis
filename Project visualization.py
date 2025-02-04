import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame (assuming it has already been processed as in the previous step)
df = pd.read_csv('processed_data.csv')

# Select the first 10 rows
data_sample = df.head(10)

# Pie Chart: Distribution of Gender
gender_counts = data_sample['Gender'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Gender Distribution (First 10 Records)')
plt.show()

# Bar Chart: Salary by Department
plt.figure(figsize=(8, 6))
sns.barplot(x='Department', y='Salary', data=data_sample, palette='muted')
plt.title('Salary by Department (First 10 Records)')
plt.xticks(rotation=45)
plt.show()

# Scatter Plot: Age vs Salary
plt.figure(figsize=(8, 6))
plt.scatter(data_sample['Age'], data_sample['Salary'], color='blue', edgecolor='k')
plt.title('Age vs Salary (First 10 Records)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Line Chart: Experience over Age
plt.figure(figsize=(8, 6))
plt.plot(data_sample['Age'], data_sample['Experience'], marker='o', linestyle='-', color='green')
plt.title('Experience over Age (First 10 Records)')
plt.xlabel('Age')
plt.ylabel('Experience (Years)')
plt.show()

# Histogram: Age Distribution
plt.figure(figsize=(8, 6))
plt.hist(data_sample['Age'], bins=5, color='orange', edgecolor='black')
plt.title('Age Distribution (First 10 Records)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
