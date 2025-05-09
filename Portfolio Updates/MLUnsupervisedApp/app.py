# importing necessary libraries, functions, and datasets
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# create a title and description with instructions at top of screen
st.title("Interactive Unsupervised Machine Learning: KMeans Clustering with Principal Component Analysis") 
st.markdown("""
### About and Instructions
            
This application is meant to be an interactive experience for students to learn about unsupervised machine learning with **KMeans CLustering** and **Principal Component Analysis (PCA)**.   
In this app you get to *upload* your own dataset and *choose* the number of clusters you would like to train on.  
The model will then be trained on the dataset provided and performance metrics will be displayed.   
A **PCA plot** will be used to visualize the clusters.
            
Instructions:  
    1 Upload your dataset in **CSV format** using the sidebar or use an example dataset of wines.  
    2 Select the numerical features you want to include in your analysis.  
    3 Choose the number of clusters (k) and a random state.  
    4 Click "Train Model" to train the model and view performance metrics.  
            """)

# create a sidebar for the data upload, feature selection, and hyperparameter selection
st.sidebar.header("1. Upload Your Dataset or Use the Wine Sample Data")
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if file:
    data = pd.read_csv(file)
else: 
    data = load_wine(as_frame=True).frame.drop(columns=['target']) # sample dataset
    
# allow the user to select the features
st.sidebar.header("2. Select Features")
features = st.sidebar.multiselect("Select features:", data.columns)
if len(features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

# allow the user to select the number of clusters and random state
st.sidebar.header("3. Select Hyperparameters")
k = st.sidebar.slider("Number of clusters", 2, 10, 3)
random_state = st.sidebar.slider("Random State", 1, 100, 23, 1)

# use a button to train the model to avoid running the model on every change
if st.sidebar.button("Train Model"):
    # drop rows with missing values in selected features
    data = data[features].dropna()  

    # standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # create a KMeans model and fit it to the data
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(data_scaled)

    # print cluster centroids 
    st.write("Cluster Centroids:")
    st.write(kmeans.cluster_centers_)

    # visualize the clusters using PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # create a scatter plot of the PCA components
    labels = sorted(np.unique(labels))
    plt.figure(figsize=(10, 6))
    for i in labels:
        plt.scatter(data_pca[kmeans.labels_ == i, 0], data_pca[kmeans.labels_ == i, 1], label=f'Cluster {i + 1}', edgecolor='k', s=50)
    plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], c='black', marker='X', s=100, label='Centroids')
    plt.title(f'KMeans Clustering with PCA (k={k})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


    # calculate silhouette score
    silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    # create elbow plot and silhouette score plot
    ks = range(2, 11)
    wcss = []
    silhouette_scores = []
    for i in ks:
        km = KMeans(n_clusters=i, random_state=random_state)
        km.fit(data_scaled)
        wcss.append(km.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled, km.labels_))

    # create elbow plot
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ks, wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)

    # create silhouette score plot
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    st.pyplot(plt)

    # use an expander to show definitions of terms used throughout the app
    with st.expander("Definitions"):
        st.subheader("Definitions")
        st.write("""
            **KMeans Clustering**: A type of unsupervised machine learning algorithm that classifies data using k amount of centroids and assigns each data point to the nearest centroid.   
            **Principal Component Analysis (PCA)**: A type of dimensionality reduction technique that transforms data from a multi-dimensional plane to an easier to visualize 2D plane by finding the two components that explain the most variance in the data.   
            **Cluster Centroids**: The average point for each cluster.   
            **Silhouette Score**: A metric used to evaluate the quality of clustering. It ranges from -1 to 1, where a higher value indicates well-defined clusters.   
            **Elbow Method**: A technique used to determine the optimal number of clusters by plotting the WCSS against the number of clusters and looking for an "elbow" point.   
            """)
        st.write("**Random State**: A value used to reproduce the same results each time the model is trained.")
        st.write("**Features**: The variables upon which the model is trained. Numeric features are recommended.")