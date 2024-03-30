from sklearn.model_selection import train_test_split

from testbed.data.utils import get_data
from testbed.data.utils import list_data

# List available datasets
datasets = list_data()


# Initialize dictionaries to store models and predictions for each dataset
models = {}
preds = {}

# Iterate over each dataset
for dataset_name in datasets:
    print(f"Processing dataset: {dataset_name}")

    # Load the dataset
    data = get_data(dataset_name, verbose=True)
    print(f"Categorical variables: {data['categorical']}")
    print("\n  ")

    X = data["x"]
    y = data["y"]

    # convert to float
    X = X.astype(float)
    y = y.astype(float)
    print(X)
    print(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data["x"], data["y"], test_size=0.2, random_state=42
    )

    # Initialize dictionaries to store models and predictions for the current dataset
    models[dataset_name] = {}
    preds[dataset_name] = {}

    ## Fit treeffuser model
    models[dataset_name]["treeffuser"] = LightGBMTreeffuser(
        verbose=1,
        n_repeats=100,
        n_estimators=10000,
        sde_name="vesde",
        learning_rate=0.05,
        early_stopping_rounds=10,
    )
    temp = models[dataset_name]["treeffuser"].fit(X_train, y_train)

    # Sample from the fitted treeffuser model
    y_samples = models[dataset_name]["treeffuser"].sample(
        X_test, n_samples=100, n_parallel=100, denoise=False, n_steps=30, seed=0
    )
    preds[dataset_name]["treeffuser"] = y_samples.mean(axis=1)

    # Fit NGBoost model with Gaussian likelihood
    models[dataset_name]["ngb"] = NGBRegressor(
        validation_fraction=0.2,
        n_estimators=10000,
        early_stopping_rounds=10,
        learning_rate=0.05,
    )
    models[dataset_name]["ngb"].fit(X_train, y_train)

    # Make predictions using the NGBoost model
    y_dim = data["y"].shape[1]
    preds[dataset_name]["ngb"] = (
        models[dataset_name]["ngb"].predict(X_test).reshape((-1, y_dim))
    )

    # Calculate and print accuracy metrics for the current dataset
    print(f"Metrics for dataset: {dataset_name}")
    metrics = compare_accuracy(preds[dataset_name], y_test, print_table=True)
    print("\n")


###########################################################
#                 CONSTANTS                               #
###########################################################

AVAILABLE_DATASETS = list_data()

AVAILABLE_MODELS = {
    "ngboost_gaussian": NGBoostGaussian,
    "ngboost_mixture_gaussian": NGBoostMixtureGaussian,
}


###########################################################
#               HELPER FUNCTIONS                          #
###########################################################


###########################################################
#               MAIN FUNCTION                            #
###########################################################


def main():
    pass


if __name__ == "__name__":
    main()
