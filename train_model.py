import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import OUTPUT_DIR

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestClassifier instance
    model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    return model  # Return the trained model

def plot_income_distribution(df):
    # Bar chart for income distribution
    income_counts = df['salary'].value_counts()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=income_counts.index.astype(str), y=income_counts.values)
    plt.title('Income Distribution')
    plt.xlabel('Income Level')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'income_distribution.png'))
    plt.close()

    # Pie chart for income distribution
    plt.figure(figsize=(8, 6))
    plt.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Income Distribution Pie Chart')
    plt.savefig(os.path.join(OUTPUT_DIR, 'income_distribution_pie.png'))
    plt.close()

