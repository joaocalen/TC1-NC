from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error


def prepare_data(ts, column='Total Renewable Energy'):
    # Select features and target variable
    X = ts.drop(['dt', column], axis=1)
    y = ts[column]

    # Split the data using TimeSeriesSplit
    time_split = TimeSeriesSplit(n_splits=4)
    train_index, test_index = list(time_split.split(X))[-1]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Percent of data in training set: {len(X_train) / len(X) * 100:.2f}%")
    print(f"Percent of data in test set: {len(X_test) / len(X) * 100:.2f}%")

    # Standardize the features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Standardization complete.")

    return X_train, y_train, X_test, y_test


def train_neural_network_bp(individual, X_train, y_train, X_test, y_test, max_hidden_neurons=50):
    # Extract hyperparameters from the individual
    hidden_neurons = int(max_hidden_neurons * (individual[0] * 10 + individual[1]) / 99)
    hidden_neurons += 1  # from [0,49] to [1,50]
    
    individual = individual.astype(int)
    learning_rate_init = 0.001 + (individual[2] * 10 + individual[3]) * (0.01 - 0.001) / 99
    learning_rate_init = min(max(learning_rate_init, 0.001), 0.01)
    
    # Create the MLPRegressor with the given hyperparameters
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_neurons,),
        learning_rate_init=learning_rate_init,
        random_state=1234,
        max_iter=1000,
    )

    # Train the model
    mlp.fit(X_train, y_train)

    # Predict and calculate RMSE
    predictions = mlp.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)

    return rmse