import numpy as np


def main():
    # import original dataset
    x = np.genfromtxt("original/CASP.csv", delimiter=",", skip_header=True)

    # extract outcome and covariates
    y = x[:, 0].copy().reshape((-1, 1))
    x = np.delete(x, 0, 1)
    categorical = []

    np.save("data.npy", {"x": x, "y": y, "categorical": categorical})


if __name__ == "__main__":
    main()
