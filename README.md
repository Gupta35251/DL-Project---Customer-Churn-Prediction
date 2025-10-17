# DL-Project---Customer-Churn-Prediction

This project develops a deep learning model to predict customer churn for a bank. 🏦

Project Description

This project focuses on building an Artificial Neural Network (ANN) to predict whether a customer is likely to leave a bank (churn). By analyzing customer data such as credit score, age, and tenure, the model provides a probability of churn, helping the bank to proactively retain customers.

Core Libraries Used

    Pandas: For loading and manipulating the customer dataset.

    Scikit-learn: For data preprocessing and model evaluation:

        LabelEncoder & OneHotEncoder: To convert categorical features into a numerical format.

        StandardScaler: To scale numerical features for better model performance.

        train_test_split: To divide the data into training and testing sets.

        metrics: To evaluate the model's performance using a confusion matrix and accuracy score.

    Keras: The deep learning library used to build the ANN:

        Sequential: To create a linear stack of layers.

        Dense: To create fully connected layers for the neural network.

Key Project Steps

    Data Preprocessing: The project begins by loading the dataset and performing one-hot encoding on categorical features like Geography and Gender. Unnecessary columns such as RowNumber, CustomerId, and Surname are dropped.

    Feature Scaling: All numerical features are scaled using StandardScaler to ensure that all variables are on a similar scale, which helps the neural network learn more effectively.

    Model Architecture: An Artificial Neural Network (ANN) is built using a sequential model in Keras. The network consists of an input layer, two hidden layers with relu activation, and an output layer with a sigmoid activation function to produce a probability score for churn.

    Model Training: The model is compiled with the adam optimizer and binary_crossentropy loss function. It is then trained on the training dataset for 100 epochs.

    Model Evaluation: The performance of the trained model is evaluated on the test set. A confusion matrix is generated to assess the model's accuracy in predicting customer churn, and the overall accuracy score is calculated.
