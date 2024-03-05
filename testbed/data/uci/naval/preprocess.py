import numpy as np


def main():
    x = np.genfromtxt("original/data.txt", delimiter="   ", skip_header=False)

    # extract outcome and covariates
    y = x[:, -1].copy().reshape((-1, 1))
    x = np.delete(x, -1, 1)
    categorical = []

    np.save("data.npy", {"x": x, "y": y, "categorical": categorical})


if __name__ == "__main__":
    main()
