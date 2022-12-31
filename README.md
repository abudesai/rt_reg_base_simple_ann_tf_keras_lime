Simple ANN in TensorFlow/Keras with LIME for Regression

- elastic net
- lime
- xai
- local explanations
- tensorflow
- keras
- regularization
- activity regularizer
- activation layer
- neural network
- sgd optimizer
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker
- abalone
- auto prices
- computer activity
- heart disease
- white wine quality
- ailerons

This is an Simple ANN Regressor, with a single hidden layer with non-linear activation.
Model applies l1 and l2 regularization. This model is built in the Tensorflow/Keras.

Artificial Neural Networks are ideal for when the data cannot be linearly separated. Perceptrons cannot handle non-linearity.

Keras supports the early stopping of training via a callback called EarlyStopping. The EarlyStopping criteria here is defined such that the neural network stops training when the log loss cannot be minimized further.

We also have a custom callback called InfCostStopCallback which stops training when cost is infinity. This can occur during hyper-parameter training when learning_rate is too high.

Model explainability is provided using LIME. Local explanations are provided here. Explanations at each instance can be understood using LIME. These explanations can be viewed by means of various plots.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT includes choosing the optimal values for learning rate for the SDG optimizer, L1 and L2 regularization and the activation function for the neural network.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include Tensorflow and Keras for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service.

The web service provides three endpoints-
/ping for health check
/infer for predictions in real time
/explain for obtaining local explanations for few samples (maximum of 5)
