import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from ngboost.distns import RegressionDistn
from ngboost.scores import LogScore
from numpy import ndarray
from sklearn.mixture import GaussianMixture as skGaussianMixture


@jax.jit
def _mixture_logpdf(
    Y: Float[ndarray, "batch 1"],
    mus: Float[ndarray, "batch k"],
    log_scales: Float[ndarray, "batch k"],
    logit_weights: Float[ndarray, "k"],
) -> Float[ndarray, "batch"]:
    """Log-probability density function of a Gaussian mixture model.

    Parameters
    ----------
    Y : Float[ndarray, "batch 1"]
        Data points.
    mus : Float[ndarray, "batch k"]
        Means of the Gaussian components.
    log_scales : Float[ndarray, "batch k"]
        Logarithm of the standard deviations of the Gaussian components. Need to be exponentiated.
    logit_weights : Float[ndarray, "k"]
        Logit of the weights of the Gaussian components. Need to be softmaxed.

    Returns
    -------
    Float[ndarray, "batch"]
        Log-probability density function of the Gaussian mixture model.
    """
    scales = jnp.exp(log_scales)
    normalized_logits: Float[ndarray, "batch k"] = logit_weights - jax.nn.logsumexp(
        logit_weights, axis=-1, keepdims=True
    )
    logprobs_per_component = jax.scipy.stats.norm.logpdf(Y, mus, scales)
    logprobs_per_component += normalized_logits
    log_probs: Float[ndarray, "batch"] = jax.nn.logsumexp(logprobs_per_component, axis=-1)
    return log_probs


_mixture_logpdf_grad_mus = jax.jit(
    jax.vmap(jax.grad(_mixture_logpdf, argnums=1), in_axes=(0, 0, 0, 0))
)
_mixture_logpdf_grad_log_scales = jax.jit(
    jax.vmap(jax.grad(_mixture_logpdf, argnums=2), in_axes=(0, 0, 0, 0))
)
_mixture_logpdf_grad_logit_weights = jax.jit(
    jax.vmap(jax.grad(_mixture_logpdf, argnums=3), in_axes=(0, 0, 0, 0))
)


def _sample_mixture(
    mus: Float[ndarray, "batch k"],
    scales: Float[ndarray, "batch k"],
    weights: Float[ndarray, "batch k"],
) -> Float[ndarray, "batch"]:
    """Sample from a Gaussian mixture model.

    Parameters
    ----------
    mus : Float[ndarray, "batch k"]
        Means of the Gaussian components.
    scales : Float[ndarray, "batch k"]
        Standard deviations of the Gaussian components.
    weights : Float[ndarray, "batch k"]
        Weights of the Gaussian components.

    Returns
    -------
    Float[ndarray, "batch"]
        Samples from the Gaussian mixture model.
    """
    component_rnd = np.random.uniform(size=(len(mus),))
    component_idx = np.argmax(np.cumsum(weights) > component_rnd[:, None], axis=1)
    return np.random.normal(
        mus[np.arange(len(mus)), component_idx], scales[np.arange(len(mus)), component_idx]
    )


class GaussianMixtureScore(LogScore):
    def score(self, Y):
        Y = Y.reshape(-1, 1)
        return -_mixture_logpdf(Y, self.mus, self.log_scales, self.logit_weights)

    def d_score(self, Y):
        """Compute the gradient of the score with respect to the parameters of the model.
        In order: mus, log_scales, logit_weights.
        """
        k = self.mus.shape[1]
        Y = Y.reshape(-1, 1)
        D = np.zeros((len(Y), 3 * k))
        # need to use vmap to compute the gradient for each data point
        D[:, :k] = _mixture_logpdf_grad_mus(Y, self.mus, self.log_scales, self.logit_weights)
        D[:, k : 2 * k] = _mixture_logpdf_grad_log_scales(
            Y, self.mus, self.log_scales, self.logit_weights
        )
        D[:, 2 * k :] = _mixture_logpdf_grad_logit_weights(
            Y, self.mus, self.log_scales, self.logit_weights
        )
        return -D


def build_gaussian_mixture_model(k: int):
    class GaussianMixture(RegressionDistn):
        """
        Implements the Mixture of Gaussians distribution.

        The Mixture of Gaussians distribution with k components has 3k parameters:
        - mus: means of the components
        - log_scales: logarithm of the standard deviations of the components
        - logit_weights: logit of the weights of the components
        """

        n_params = 3 * k
        scores = (GaussianMixtureScore,)

        def __init__(self, params):
            super().__init__(params)
            self.mus = params[0:k].T
            self.log_scales = params[k : 2 * k].T
            self.logit_weights = params[2 * k :].T
            self.scale = np.exp(self.log_scales)
            self.var = self.scale**2
            self.weights = jax.nn.softmax(self.logit_weights, axis=-1)

        def fit(Y):
            skm = skGaussianMixture(n_components=k)
            skm.fit(Y[:, None])
            mus = skm.means_[:, 0]
            log_scales = np.log(skm.covariances_)[:, 0, 0] / 2
            logit_weights = np.log(skm.weights_)
            return np.concatenate([mus, log_scales, logit_weights])

        def sample(self, m):
            return np.array(
                [_sample_mixture(self.mus, self.scale, self.weights) for _ in range(m)]
            )

        # @property
        # def params(self):
        #     return {"loc": self.loc, "scale": self.scale}

        def mean(self):
            return np.sum(self.mus * self.weights, axis=1)

    return GaussianMixture
