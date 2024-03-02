import numpy as np


def main():
    x = np.genfromtxt("original/yacht_hydrodynamics.data", skip_header=False)

    # extract outcome and covariates
    y = x[:, -1].copy()
    x = np.delete(x, -1, 1)
    categorical = []

    np.save("data.npy", {"x": x, "y": y, "categorical": categorical})


if __name__ == "__main__":
    main()
