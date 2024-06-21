import numpy as np
import pytest

from treeffuser.samples import Samples


###################################################
# Fixtures for random samples
###################################################
@pytest.fixture(scope="module")
def seed():
    return 0


@pytest.fixture(scope="module")
def normal_samples(seed):
    rng = np.random.default_rng(seed=seed)
    n_samples = 10**4
    loc = 1
    scale = np.sqrt(2)
    return {
        "samples": rng.normal(loc=loc, scale=scale, size=(n_samples, 15, 1)),
        "loc": loc,
        "scale": scale,
    }


@pytest.fixture(scope="module")
def uniform_samples(seed):
    rng = np.random.default_rng(seed=seed)
    n_samples = 10**4
    low = 0
    high = 1
    return {
        "samples": rng.uniform(low=low, high=high, size=(n_samples, 15, 1)),
        "low": low,
        "high": high,
    }


@pytest.fixture(scope="module")
def multivariate_normal_samples(seed):
    rng = np.random.default_rng(seed=seed)
    n_samples = 10**3
    batch = 15
    y_dim = 3
    mean = np.zeros(y_dim)
    cov = np.array([[1, 0.8, 0.6], [0.8, 1, 0.8], [0.6, 0.8, 1]])

    return {
        "samples": rng.multivariate_normal(mean, cov, size=(n_samples, batch)),
        "mean": mean,
        "cov": cov,
    }


###################################################
# Shapes and subscripting
###################################################
def test_samples_basic(multivariate_normal_samples):
    samples = Samples(multivariate_normal_samples["samples"])
    _, batch, y_dim = multivariate_normal_samples["samples"].shape

    assert samples.sample_correlation().shape == (batch, y_dim, y_dim)
    assert len(samples.sample_kde()) == batch
    assert samples.sample_max().shape == (batch, y_dim)
    assert samples.sample_mean().shape == (batch, y_dim)
    assert samples.sample_median().shape == (batch, y_dim)
    assert samples.sample_min().shape == (batch, y_dim)
    assert samples.sample_quantile(q=[0.05, 0.95]).shape == (2, batch, y_dim)
    assert samples.sample_std().shape == (batch, y_dim)

    with pytest.raises(ValueError, match="only applies to unidimensional responses"):
        samples.sample_confidence_interval()
    with pytest.raises(ValueError, match="only applies to unidimensional responses"):
        assert len(samples.sample_mode()) == batch
    with pytest.raises(ValueError, match="only applies to unidimensional responses"):
        assert samples.sample_range().shape == (batch, y_dim, 2)

    assert samples[..., 0].sample_confidence_interval().shape == (2, batch)
    assert samples[..., 0].sample_mode().shape == (batch,)
    assert samples[..., 0].sample_range().shape == (batch, 2)


def test_samples_subscript(multivariate_normal_samples):
    samples = Samples(multivariate_normal_samples["samples"])
    n_samples, batch, y_dim = multivariate_normal_samples["samples"].shape

    # check that an error is raised when the first or the second dimensions are subscripted
    # with an integer
    with pytest.raises(ValueError, match="first dimension of the samples"):
        assert samples[0].shape == (batch, y_dim)
    with pytest.raises(ValueError, match="second dimension of the samples"):
        assert samples[:, 0].shape == (n_samples, y_dim)

    # check that the subscripting works as expected otherwise
    assert samples[..., 0].shape == (n_samples, batch)
    assert samples[:, [0]].shape == (n_samples, 1, y_dim)
    assert samples[[0], ...].shape == (1, batch, y_dim)
    assert samples[1:3, 2:4, 1].shape == (2, 2)


###################################################
# Methods
###################################################
def test_samples_apply(normal_samples):
    def kurtosis(samples):
        return np.mean((samples - np.mean(samples)) ** 4) / (np.std(samples) ** 4)

    samples = Samples(normal_samples["samples"])
    _, batch, y_dim = normal_samples["samples"].shape

    true_kurtosis = 3
    sample_kurtosis = samples.sample_apply(fun=kurtosis)

    assert sample_kurtosis.shape == (batch, y_dim)
    for i in range(batch):
        assert np.allclose(sample_kurtosis[i, :], true_kurtosis, atol=0.1)


def test_samples_confidence_interval_and_quantiles(normal_samples):
    samples = Samples(normal_samples["samples"])
    conditional_loc = normal_samples["loc"]
    conditional_scale = normal_samples["scale"]
    true_q_975 = conditional_loc + 1.96 * conditional_scale
    true_q_025 = conditional_loc - 1.96 * conditional_scale
    true_confidence_interval = [true_q_025, true_q_975]

    sample_q_975 = samples.sample_quantile(q=0.975)
    sample_q_025 = samples.sample_quantile(q=0.025)
    sample_confidence_interval = samples.sample_confidence_interval(confidence=0.95)

    assert sample_q_975.shape == (1, samples.batch, samples.y_dim)
    assert sample_q_025.shape == (1, samples.batch, samples.y_dim)
    assert sample_confidence_interval.shape == (2, samples.batch, samples.y_dim)
    for i in range(samples.batch):
        assert np.allclose(sample_q_975[:, i, :], true_q_975, atol=0.1)
        assert np.allclose(sample_q_025[:, i, :], true_q_025, atol=0.1)
        assert np.allclose(
            sample_confidence_interval[:, i, :].reshape(-1), true_confidence_interval, atol=0.1
        )


def test_samples_correlation(multivariate_normal_samples):
    samples = Samples(multivariate_normal_samples["samples"])
    batch = multivariate_normal_samples["samples"].shape[1]

    true_correlation = multivariate_normal_samples["cov"]  # stds are equal to 1
    sample_correlation = samples.sample_correlation()

    for i in range(batch):
        assert np.allclose(sample_correlation[i, :, :], true_correlation, atol=0.1)


@pytest.mark.parametrize(
    "statistic, true_value",
    [
        ("sample_mean", 1),
        ("sample_median", 1),
        ("sample_mode", 1),
        ("sample_std", np.sqrt(2)),
    ],
)
def test_samples_main_statistics(statistic, true_value, normal_samples):
    samples = Samples(normal_samples["samples"])
    batch = normal_samples["samples"].shape[1]

    sample_stat = getattr(samples, statistic)()
    for i in range(batch):
        assert np.allclose(sample_stat[i, ...], true_value, atol=0.1)


@pytest.mark.parametrize(
    "statistic, true_value",
    [
        ("sample_max", 1),
        ("sample_min", 0),
    ],
)
def test_samples_max_min(statistic, true_value, uniform_samples):
    samples = Samples(uniform_samples["samples"])
    batch = uniform_samples["samples"].shape[1]

    sample_stat = getattr(samples, statistic)()
    for i in range(batch):
        assert np.allclose(sample_stat[i, ...], true_value, atol=0.1)
