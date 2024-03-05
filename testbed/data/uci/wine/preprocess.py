import numpy as np


def main():
    # import original datasets
    red = np.genfromtxt("original/winequality-red.csv", delimiter=";", skip_header=True)
    white = np.genfromtxt("original/winequality-white.csv", delimiter=";", skip_header=True)

    # concatenate them
    x = np.concatenate((red, white), axis=0)

    # add covariate for red vs. white
    red_color = np.array([1] * red.shape[0] + [0] * white.shape[0], dtype=np.float64).reshape(
        (-1, 1)
    )
    x = np.concatenate((x, red_color), axis=1)

    # extract outcome and covariates
    y = x[:, -2].copy().reshape((-1, 1))
    x = np.delete(x, -2, 1)
    categorical = [x.shape[1]]

    np.save("data.npy", {"x": x, "y": y, "categorical": categorical})


if __name__ == "__main__":
    main()
