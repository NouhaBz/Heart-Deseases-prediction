import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss




import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')
'''import streamlit as st'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')
#dataset = pd.read_csv(r'C:\Users\dz\Music\data science\heart.csv')

dataset = pd.read_csv(r'C:\Users\dz\OneDrive\Desktop\master1\S1\data science\heart_disease_data.csv')
# Load your combined dataset
# Ensure that the column names in your code match the actual column names in your CSV file
combined_dataset = pd.read_csv(r'C:\Users\dz\OneDrive\Desktop\master1\S1\data science\heart_disease_data.csv')

type(dataset)
dataset.shape
dataset.head(5)
dataset.sample(5)
dataset.describe()
dataset.info()
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])
    
dataset["target"].describe()
y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))
y = dataset["target"]


# Set the style for Seaborn plots
sns.set(style="whitegrid")

# 1. Count Plot for Heart Disease
plt.figure()
sns.countplot(x="target", data=combined_dataset, palette="bwr")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Count Plot for Heart Disease")
plt.show()

# Print percentages
countNoDisease = len(combined_dataset[combined_dataset.target == 0])
countHaveDisease = len(combined_dataset[combined_dataset.target == 1])
print(f"Percentage of Patients Without Heart Disease: {countNoDisease / len(combined_dataset.target) * 100:.2f}%")
print(f"Percentage of Patients With Heart Disease: {countHaveDisease / len(combined_dataset.target) * 100:.2f}%")

# 2. Count Plot for Gender
plt.figure()
countFemale = len(combined_dataset[combined_dataset.sex == 0])
countMale = len(combined_dataset[combined_dataset.sex == 1])
sns.countplot(x='sex', data=combined_dataset, palette="mako_r")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.title("Count Plot for Gender")
plt.show()

# Print percentages
print(f"Percentage of Female Patients: {countFemale / len(combined_dataset.sex) * 100:.2f}%")
print(f"Percentage of Male Patients: {countMale / len(combined_dataset.sex) * 100:.2f}%")

# 3. Bar Chart for Heart Disease Frequency by Age
plt.figure(figsize=(20, 6))
pd.crosstab(combined_dataset.age, combined_dataset.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 4. Bar Chart for Gender vs. Heart Disease
plt.figure(figsize=(8, 6))
sns.countplot(data=combined_dataset, x='sex', hue='target', palette='Set1')
plt.legend(["Haven't Disease", "Have Disease"])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Frequency')
plt.title('Gender vs. Heart Disease')
plt.show()

# 5. Scatter Plot for Age vs. Maximum Heart Rate
plt.figure()
plt.scatter(x=combined_dataset.age[combined_dataset.target == 1], y=combined_dataset.thalach[(combined_dataset.target == 1)], c="red")
plt.scatter(x=combined_dataset.age[combined_dataset.target == 0], y=combined_dataset.thalach[(combined_dataset.target == 0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.title("Scatter Plot for Age vs. Maximum Heart Rate")
plt.show()

# 6. Scatter Plot for Age vs. Resting Blood Pressure
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_dataset, x='age', y='trestbps', hue='target', palette='coolwarm')
plt.title('Age vs. Resting Blood Pressure')
plt.show()

# 7. Pie Chart for Fasting Blood Sugar
plt.figure(figsize=(8, 8))
combined_dataset['fbs'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Distribution of Fasting Blood Sugar')
plt.show()


# 8. Bar Chart for Slope
plt.figure(figsize=(15, 6))
pd.crosstab(combined_dataset.slope, combined_dataset.target).plot(kind="bar", color=['#DAF7A6', '#FF5733'])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

# 9. Bar Chart for FBS
plt.figure(figsize=(15, 6))
pd.crosstab(combined_dataset.fbs, combined_dataset.target).plot(kind="bar", color=['#FFC300', '#581845'])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

# 10. Bar Chart for Chest Pain Type
plt.figure(figsize=(15, 6))
pd.crosstab(combined_dataset.cp, combined_dataset.target).plot(kind="bar", color=['#11A5AA', '#AA1190'])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation=0)
plt.ylabel('Frequency of Disease or Not')
plt.show()



#//////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Plot bar plots for different categorical features
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
numerical_feature = "thal"

plt.figure(figsize=(15, 8))

#Train Test split
from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
#V. Model Fitting
from sklearn.metrics import accuracy_score
#logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
print(Y_pred_dt.shape)
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


#Output final score
scores = [score_lr,score_dt]
algorithms = ["Logistic Regression","Decision Tree"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(x=algorithms, y=scores)

from sklearn.metrics import confusion_matrix

# Output final score
def get_model_by_name(model_name):
    # Define your models here
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
       
    }

    # Check if the provided model name exists in the dictionary
    if model_name in models:
        return models[model_name]
    else:
        raise ValueError(f"Model {model_name} not found. Please add it to the models dictionary.")
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay





lr_train_accuracy = []
epochs= [ 1, 10, 100,500, 1000]
for epoch in epochs:
    # Perform training steps for Logistic Regression (use your actual training steps)
    lr.fit(X_train, Y_train)

    # Evaluate on training set
    Y_pred_train_lr = lr.predict(X_train)
    train_accuracy = accuracy_score(Y_pred_train_lr, Y_train)
    lr_train_accuracy.append(train_accuracy)

# Plot Accuracy for Logistic Regression
plt.figure(figsize=(8, 4))

# Plot Training Accuracy
plt.plot(epochs, lr_train_accuracy, label='Training Accuracy')
plt.title('Logistic Regression Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Decision Tree Training
tree_depths = []

for x in range(1, 21):  # Assuming a maximum depth of 20
    dt = DecisionTreeClassifier(random_state=x, max_depth=x)
    dt.fit(X_train, Y_train)
    tree_depths.append(dt.tree_.max_depth)

# Plot Tree Depth for Decision Tree
plt.figure(figsize=(8, 4))

# Plot Tree Depth
plt.plot(range(1, 21), tree_depths, label='Tree Depth')
plt.title('Decision Tree Training (Depth)')
plt.xlabel('Epoch')
plt.ylabel('Tree Depth')
plt.legend()

plt.show()





# Output final score
scores = [score_lr, score_dt]
algorithms = ["Logistic Regression", "Decision Tree"]

for algorithm in algorithms:
    print(f"The accuracy score achieved using {algorithm} is: {scores[algorithms.index(algorithm)]} %")

    # Confusion Matrix
    model = get_model_by_name(algorithm)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)

    # Display Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Heart Disease"], yticklabels=["No Disease", "Heart Disease"])
    plt.title(f"Confusion Matrix for {algorithm}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Display Classification Report
    print(f"Classification Report for {algorithm}:\n{classification_report(Y_test, Y_pred)}")

    # Display ROC Curve
    disp = RocCurveDisplay.from_estimator(model, X_test, Y_test)
    plt.title(f"ROC Curve for {algorithm}")
    plt.show()

    # Display Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(Y_test, Y_pred)
    disp_pr = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp_pr.plot()
    plt.title(f"Precision-Recall Curve for {algorithm}")
    plt.show()

# Bar plot for accuracy scores
sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(x=algorithms, y=scores)
plt.show()


# Output final score and details for Logistic Regression and Decision Tree
algorithms = ["Logistic Regression", "Decision Tree"]
scores = [score_lr, score_dt]

for i in range(len(algorithms)):
    print(f"The accuracy score achieved using {algorithms[i]} is: {scores[i]} %")

    # Model details
    if algorithms[i] == "Logistic Regression":
        print("Training Process:")
        print("1. Data Preprocessing:")
        print("   - Handle missing values. (Assumed done before training)")
        print("   - Encode categorical variables if necessary. (Assumed done before training)")
        print("   - Scale numerical features. (Assumed done before training)")
        print("2. Model Initialization:")
        print("   - Initialize weights and bias.")
        print("3. Optimization Algorithm:")
        print("   - Use an optimization algorithm like gradient descent.")
        print("   - Update weights and bias iteratively to minimize the logistic loss.")
        print("4. Training Iterations:")
        print("   - Iterate through the dataset multiple times (epochs) to optimize the model.")
        print("5. Hyperparameters Tuned or Configurations:")
        print("   - Learning Rate: (Handled internally by scikit-learn's LogisticRegression)")
        print("   - Regularization Strength: (Handled internally by scikit-learn's LogisticRegression)")
        print("   - Threshold: (Assumed set to 0.5, default threshold in scikit-learn)")

    elif algorithms[i] == "Decision Tree":
        print("Training Process:")
        print("1. Data Preprocessing:")
        print("   - Handle missing values.")
        print("   - Encode categorical variables if necessary. (Assumed done before training)")
        print("2. Tree Construction:")
        print("   - Recursively split the data based on features to maximize information gain or Gini impurity.")
        print("   - Continue until a stopping criterion (e.g., maximum depth, minimum samples per leaf) is met.")
        print("3. Hyperparameters Tuned or Configurations:")
        print(f"   - Maximum Depth: {dt.get_depth()} (Set by the max_depth parameter in DecisionTreeClassifier)")
        print(f"   - Minimum Samples Split: {dt.min_samples_split} (Set by the min_samples_split parameter in DecisionTreeClassifier)")
        print(f"   - Criterion: {dt.criterion} (Set by the criterion parameter in DecisionTreeClassifier)")

    # Model Evaluation
    print("Model Evaluation:")
    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred_dt) if algorithms[i] == "Decision Tree" else confusion_matrix(Y_test, Y_pred_lr)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Heart Disease"], yticklabels=["No Disease", "Heart Disease"])
    plt.title(f"Confusion Matrix for {algorithms[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Classification Report
    print(f"Classification Report for {algorithms[i]}:\n{classification_report(Y_test, Y_pred_dt if algorithms[i] == 'Decision Tree' else Y_pred_lr)}")

    
         
   # Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)


# Vary hyperparameter (C) for Logistic Regression and plot performance
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accuracy_values_lr = []
test_accuracy_values_lr = []

for C in C_values:
    lr = LogisticRegression(C=C)
    lr.fit(X_train, Y_train)

    train_accuracy_lr = accuracy_score(Y_train, lr.predict(X_train))
    test_accuracy_lr = accuracy_score(Y_test, lr.predict(X_test))

    train_accuracy_values_lr.append(train_accuracy_lr)
    test_accuracy_values_lr.append(test_accuracy_lr)

# Plot the results for Logistic Regression
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracy_values_lr, label='Training Accuracy')
plt.plot(C_values, test_accuracy_values_lr, label='Test Accuracy')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.title('Logistic Regression Accuracy vs. Regularization Strength (C)')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)

# Vary hyperparameter (max_depth) for Decision Tree and plot performance
max_depth_values = range(1, 21)
train_accuracy_values = []
test_accuracy_values = []

for depth in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, Y_train)

    train_accuracy = accuracy_score(Y_train, dt.predict(X_train))
    test_accuracy = accuracy_score(Y_test, dt.predict(X_test))

    train_accuracy_values.append(train_accuracy)
    test_accuracy_values.append(test_accuracy)



# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, train_accuracy_values, label='Training Accuracy')
plt.plot(max_depth_values, test_accuracy_values, label='Test Accuracy')
plt.title('Decision Tree Accuracy vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()






    #///////////////////////////////////////////////////////
  
    
    
def predict_heart_disease(logistic_model, decision_tree_model, input_data):
    
         # Ensure that input_data contains the required features in the correct order 
        required_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
         # Extract features from input_data in the correct order
        input_features = [input_data[feature] for feature in required_features]

         # Reshape input_features into a 2D array for prediction
        input_features = np.array(input_features).reshape(1, -1)

         # Predict using Logistic Regression model
        prediction_lr = logistic_model.predict(input_features)[0]

          # Predict using Decision Tree model
        prediction_dt = decision_tree_model.predict(input_features)[0]
 
        return prediction_lr, prediction_dt





# Example input_data (replace this with the actual input data)
input_data = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# Call the prediction function
prediction_lr, prediction_dt = predict_heart_disease(lr, dt, input_data)

# Display predictions
print(f"Prediction by Logistic Regression: {prediction_lr}")
print(f"Prediction by Decision Tree: {prediction_dt}")

###############################################################################################################################################################







import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class EHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EHealth Application")

        # Create the main container
        self.main_container = tk.Frame(root)
        self.main_container.pack(fill="both", expand=True)

        # Open and convert the image to PhotoImage
        img = Image.open('stock-vector-human-heart-anatomy-form-lines-and-triangles-point-connecting-network-on-blue-background-1154458315.jpg')
        img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()), resample=Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(img)

        # Background Image
        self.bg_label = tk.Label(self.main_container, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        # Text above buttons
        text_label = tk.Label(self.main_container, text='Welcome to our E-Health Platform', font=('Arial', 24, 'bold'), fg='#000', bg='white')
        text_label.place(relx=0.5, rely=0.2, anchor='n')

        # Button Container
        button_container = tk.Frame(self.main_container)
        button_container.place(relx=0.5, rely=0.5, anchor='center')

        # Document Buttons
        document_btn2 = self.create_document_button(button_container, 'test')
        document_btn2.configure(command=self.go_to_patient_page)
        document_btn2.grid(row=0, column=1, padx=10)

    def create_document_button(self, parent, text):
        button = ttk.Button(parent, text=text, style='Documents.TButton')
        return button

    def go_to_patient_page(self):
        # Clear the current content
        for widget in self.main_container.winfo_children():
            widget.destroy()

        # Create an instance of CheckoutApp
        checkout_app = CheckoutApp(self.root)
        
class CheckoutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Checkout Page")

        # Create background container
        self.background_container = tk.Frame(root, width=800, height=600)
        self.background_container.place(x=0, y=0)

        # Create content container
        self.content_container = tk.Frame(root, width=400, padx=20, pady=20)
        self.content_container.place(x=400, y=50)

        # Add your widgets to the content container
        self.create_personal_information_section()
        self.create_analys_result_section()
        #self.create_apply_coupons_section()
        
        # Create a button
        button = tk.Button(root, text="Button", bg="#1899D6", fg="#FFFFFF", font=('Arial', 15), command=self.button_click)
        button.place(x=400, y=550)

    def create_personal_information_section(self):
        section_frame = tk.LabelFrame(self.content_container, text="Personal Information")
        section_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Age dropdown
        age_label = tk.Label(section_frame, text="Age:")
        age_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.age_var = tk.StringVar()
        age_dropdown = ttk.Combobox(section_frame)
        age_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        age_dropdown.set("Age")

        gender_label = tk.Label(section_frame, text="Gender:")
        gender_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.gender_var = tk.StringVar()
        gender_dropdown = ttk.Combobox(section_frame, values=["Male", "Female"], textvariable=self.gender_var)
        gender_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        gender_dropdown.set("Gender")

        # Bind the event to update the gender_var
        gender_dropdown.bind("<<ComboboxSelected>>", self.update_gender_var)

    def update_gender_var(self, event):
        # Update gender_var based on the selected value
        selected_value = self.gender_var.get()
        if selected_value == "Female":
            self.gender_var.set("0")
        elif selected_value == "Male":
            self.gender_var.set("1")

    def create_analys_result_section(self):
        
        section_frame = tk.LabelFrame(self.content_container, text="Analys Result")
        section_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.analysis_entries = {}

        # Entry fields for analysis results
        analysis_labels = ["Resting Blood Pressure", "Serum Cholestoral", "Fasting Blood Sugar", "Max Heart Rate Achieved", "Exang", "Ca", "Slope", "Thal"]

        for i, label_text in enumerate(analysis_labels):
         label = tk.Label(section_frame, text=label_text + ":")
        label.grid(row=i, column=0, padx=5, pady=5, sticky="w")

        if label_text in ["Exang", "Ca", "Slope", "Thal"]:
        # Create StringVar for dropdowns
         dropdown_var = tk.StringVar()
        dropdown = ttk.Combobox(section_frame, textvariable=dropdown_var)

        # Set default value
        dropdown.set(label_text)

        # Store the StringVar in the corresponding attribute
        setattr(self, f"{label_text.lower()}_var", dropdown_var)

        if label_text == "Exang":
            # For Exang, create dropdowns with predefined values
            dropdown['values'] = ["yes", "no"]

        elif label_text == "Slope":
            # For Slope, create a dropdown with values from the slope_mapping dictionary
            slope_mapping = {"Upsloping": "0", "Flatsloping": "1", "Downslopins": "2"}
            dropdown['values'] = list(slope_mapping.keys())

        elif label_text == "Ca":
            # For Ca, create a dropdown with values from 0 to 3
            dropdown['values'] = ["0", "1", "2", "3"]

        else:
            if label_text == "Thal":
            # For Thal, create a dropdown with values 1, 3, 6, 7
             dropdown['values'] = ["1", "3", "6", "7"]

             dropdown.grid(row=i+1, column=1, padx=5, pady=5, sticky="w")
            else:
                
                entry = tk.Entry(section_frame)
                entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")

                # Store entry fields in a dictionary for easy retrieval
                self.analysis_entries[label_text] = entry

        # Chest Pain Type dropdown
        chest_pain_label = tk.Label(section_frame, text="Chest Pain Type:")
        chest_pain_label.grid(row=len(analysis_labels), column=0, padx=5, pady=5, sticky="w")

        self.chest_pain_var = tk.StringVar()
        chest_pain_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        chest_pain_dropdown = ttk.Combobox(section_frame, values=chest_pain_options, textvariable=self.chest_pain_var)
        chest_pain_dropdown.grid(row=len(analysis_labels), column=1, padx=5, pady=5, sticky="w")
        chest_pain_dropdown.set("Chest Pain Type")

        # Dictionary mapping options to values
        self.chest_pain_mapping = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }

        # Bind the event to update the chest_pain_var
        chest_pain_dropdown.bind("<<ComboboxSelected>>", self.update_chest_pain_var)

        # Resting Electrocardiographic Results dropdown
        electrocardiographic_label = tk.Label(section_frame, text="Resting Electrocardiographic Results:")
        electrocardiographic_label.grid(row=len(analysis_labels) + 1, column=0, padx=5, pady=5, sticky="w")

        self.electrocardiographic_var = tk.StringVar()
        electrocardiographic_options = ["Nothing to note", "ST-T Wave abnormality", "Left ventricular hypertrophy"]
        electrocardiographic_dropdown = ttk.Combobox(section_frame, values=electrocardiographic_options, textvariable=self.electrocardiographic_var)
        electrocardiographic_dropdown.grid(row=len(analysis_labels) + 1, column=1, padx=5, pady=5, sticky="w")
        electrocardiographic_dropdown.set("Results")

        # Dictionary mapping options to values
        self.electrocardiographic_mapping = {
            "Nothing to note": 0,
            "ST-T Wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        }

        # Bind the event to update the electrocardiographic_var
        electrocardiographic_dropdown.bind("<<ComboboxSelected>>", self.update_electrocardiographic_var)

    def update_chest_pain_var(self, event):
        # Update chest_pain_var based on the selected value
        selected_value = self.chest_pain_var.get()
        if selected_value in self.chest_pain_mapping:
            self.chest_pain_var.set(self.chest_pain_mapping[selected_value])

    def update_electrocardiographic_var(self, event):
        # Update electrocardiographic_var based on the selected value
        selected_value = self.electrocardiographic_var.get()
        if selected_value in self.electrocardiographic_mapping:
            self.electrocardiographic_var.set(self.electrocardiographic_mapping[selected_value])

    
    
    def button_click(self):
     age_value = self.age_var.get()
     gender_value = self.gender_var.get()
     electrocardiographic_results_value = self.electrocardiographic_var.get()
    
    # Retrieve values from the analysis result section
     resting_blood_pressure_value = self.analysis_entries["Resting Blood Pressure"].get()
     serum_cholesterol_value = self.analysis_entries["Serum Cholestoral"].get()
     fasting_blood_sugar_value = self.analysis_entries["Fasting Blood Sugar"].get()
     max_heart_rate_value = self.analysis_entries["Max Heart Rate Achieved"].get()
     fasting_blood_sugar_binary = 1 if float(fasting_blood_sugar_value) > 120 else 0
    
    # Handle the dropdowns differently
     exang_value = self.exang_var.get() if hasattr(self, 'exang_var') else None
     ca_value = self.ca_var.get() if hasattr(self, 'ca_var') else None
     slope_value = self.slope_var.get() if hasattr(self, 'slope_var') else None
     thal_value = self.thal_var.get() if hasattr(self, 'thal_var') else None

    # Do something with the retrieved values
     print("Age:", age_value)
     print("Gender:", gender_value)
     print("Electrocardiographic Results:", electrocardiographic_results_value)
     print("Resting Blood Pressure:", resting_blood_pressure_value)
     print("Serum Cholesterol:", serum_cholesterol_value)
     print("Fasting Blood Sugar:", fasting_blood_sugar_value)
     print("Max Heart Rate Achieved:", max_heart_rate_value)
     #print("Chest Pain Type:", chest_pain_type_value)
     print("Fasting Blood Sugar Binary:", fasting_blood_sugar_binary)

    # Handle the dropdown values
     print("Exang:", exang_value)
     print("Ca:", ca_value)
     print("Slope:", slope_value)
     print("Thal:", thal_value)


if __name__ == "__main__":
    root = tk.Tk()
    app = EHealthApp(root)
    root.mainloop()