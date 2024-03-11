import numpy as np

from treeffuser.sde import sdeint
from treeffuser.sde.base_sde import CustomSDE


def test_euler_maruyama():
    # Samples from this SDE are ~ Normal(y, t)
    sde = CustomSDE(drift_fn=lambda y, t: 0, diffusion_fn=lambda y, t: 1)

    batch_size = 1000
    y0 = np.zeros((batch_size, 1))
    samples = sdeint(sde, y0, 0, 1, n_steps=20, method="euler", seed=0)
    assert samples.shape == (batch_size, 1)
    samples = sdeint(sde, y0, 0, 1, n_steps=20, n_samples=3, method="euler", seed=0)
    assert samples.shape == (3, batch_size, 1)

    for t in [0.5, 1, 2]:
        samples = sdeint(sde, y0, 0, t, n_steps=20, method="euler", seed=0)
        # Mean should be close to 0.
        assert np.abs(samples.mean()) < 0.1
        # Variance should be close to t.
        assert np.abs(samples.var() - t) < 0.1
