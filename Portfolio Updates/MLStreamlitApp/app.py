# import necessary processing and visualization libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import necessary machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# borrowed CM plotting from class notes
def plot_confusion_matrix(cm, title = "Confusion Matrix"):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

# create a title and description with instructions at top of screen
st.title("Interactive Machine Learning: Decision Tree vs K Nearest Neighbor Classifiers")
st.markdown("""
### About and Instructions
            
This application is meant to be an interactive experience for students to learn about **Decision Tree** and **K Nearest Neighbor** Classifiers.   
In this app you get to *upload* your own dataset, *finetune* hyperparameters, and *choose* your model.  
The model will then be trained on the dataset provided and performance metrics will be displayed.   
            
Instructions:  
    1 Upload your dataset in **CSV format** using the sidebar.  
    2 Select the target variable from the dropdown menu and the features you want to train on.   
    3 Choose the model you want to use (Decision Tree or K Nearest Neighbor).  
    4 Adjust the hyperparameters for the selected model.  
    5 Click "Train Model" to train the model and view performance metrics.  
            """)


# using a sidebar for the data upload, target selection, and model selection
# if they don't upload a file, we will use the Titanic dataset as a default
st.sidebar.header("1. Upload Your Dataset or Use Titanic Sample Data")
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if file:
    data = pd.read_csv(file)

else: 
    data = sns.load_dataset('titanic')

# Allow the user to select the target variable and features    
st.sidebar.header("2. Select Target and Features")
target = st.sidebar.selectbox("Select a ***categorical*** target variable:", data.columns)
features = st.sidebar.multiselect("Select features:", data.columns.drop(target))

# Allow the user to select the model type, test size, and random state
st.sidebar.header("3. Select Model, Test Size, and Random State")
model_type = st.sidebar.selectbox("Select a model:", ["Decision Tree", "KNN"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
random_state = st.sidebar.slider("Random State", 1, 100, 23, 1)

# Allow user to select hyperparameters for the model
scaled = False
if model_type == "Decision Tree":
    # Hyperparameters for Decision Tree
    st.sidebar.header("4. Decision Tree Hyperparameters")
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, 1)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 4, 1)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 2, 1)
    # Create the model
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

elif model_type == "KNN":
    # Hyperparameters for KNN
    st.sidebar.header("4. KNN Hyperparameters")
    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 19, 5, 2)
    scaled = st.sidebar.checkbox("Scale Features (Recommended for Accurate Analysis)")

    # Create the model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)


# Use a button to train the model so it happens all at once and doesn't show an error message
if st.button("Train Model"):
    if not features:
        st.warning("***Please select at least one feature before training.***")
        st.stop()

    # Drop missing data from selected variables, encode categorical variables, and split the data into training and testing sets
    data_selected = data[features + [target]].dropna()
    X = data_selected[features]
    X = pd.get_dummies(X, drop_first=True)

    # Scale the features if selected
    if model_type == "KNN" and scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = data_selected[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, f"{model_type} Confusion Matrix")

    # Display classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df)

    # use an expander to give a preview of the data after variable selection
    with st.expander("View your data subset"):
        st.subheader("Data Preview")
        st.dataframe(data_selected.head(10))
        st.subheader("Data Description")
        st.write(data_selected.describe())
        st.subheader("Data Types")
        st.write(data_selected.dtypes)

# use a global expander to give definitions of some important terms
with st.expander("Definitions"):
    st.subheader("Definitions")
    st.write("""
        - **Accuracy**: The percentage of observations classified correctly by the model.
        - **Precision**: The ratio of true positive predictions to the total predicted positives. Tells us how good we are at predicting positives correctly.
        - **Recall**: The ratio of true positive predictions to the total actual positives. Tells us how good we are at identifying all positive cases.
        - **Confusion Matrix**: A table used to describe the performance of a classification model. It shows the true vs predicted classifications.
        - **Classification Report**: A report that shows the main classification metrics such as precision and recall. 
        """)

