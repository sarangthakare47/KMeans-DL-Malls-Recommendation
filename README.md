Mall Clustering and Classification with ML + DL
Project Overview

This project combines unsupervised learning (ML) and supervised learning (DL) to analyze and classify malls based on various features. It first clusters malls using KMeans, then trains an Artificial Neural Network (ANN) to predict cluster categories for new malls.

Features considered:

Number of shops

Number of brands

Presence of food courts or cinema

Average ratings

Number of luxury shops

Footfall

Technologies Used

Python

Pandas, NumPy

scikit-learn (for clustering and metrics)

TensorFlow / Keras (for ANN)

How It Works
1. KMeans Clustering (ML)

Groups malls into clusters: Premium, Standard, Budget

Clustering is based solely on numeric features

Cluster labels are used as pseudo-targets for ANN

2. Artificial Neural Network (DL)

Input Layer → takes scaled features

Hidden Layers → learn patterns in the data

Output Layer → predicts cluster category

Uses ReLU activation in hidden layers and Softmax in the output

Optimizer: Adam, Loss function: Sparse Categorical Crossentropy

3. Evaluation

Confusion Matrix → shows correctly and incorrectly classified malls

Classification Report → precision, recall, F1-score per category

By - Sarang A Thakare
