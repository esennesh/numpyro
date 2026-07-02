# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Variational Autoencoder
================================
"""

import argparse
import functools
import inspect
import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

import flax.typing as flaxtyping
from jax import jit, lax, random
import jax.numpy as jnp
from jax.random import key
from flax import nnx

import numpyro
from numpyro import optim
import numpyro.contrib.diag_sgd as dsgd
from numpyro.contrib.module import nnx_module
import numpyro.distributions as dist
from numpyro.examples.datasets import MNIST, load_dataset
from numpyro.infer import SVI, Trace_ELBO

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(inspect.getfile(lambda: None)), ".results")
)
os.makedirs(RESULTS_DIR, exist_ok=True)


class PVaeEncoder(nnx.Module):
    def __init__(self, z_dim, *, rngs: nnx.Rngs, x_dim=28):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), strides=2, padding=1,
                              rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=2, padding=1,
                              rngs=rngs)
        feature_area = (x_dim // 4) ** 2
        self.linear = nnx.Linear(32 * feature_area, z_dim, rngs=rngs)

    def __call__(self, xs, rngs=None):
        hs = nnx.swish(self.conv1(xs.swapaxes(-3, -1)))
        hs = nnx.swish(self.conv2(hs))
        return self.linear(hs.reshape(hs.shape[0], -1))

class NonnegativeParam(nnx.Param):
    def get_value(self, *, index=flaxtyping.MISSING):
        value = super().get_value(index=index)
        return value ** 2

class NMFDecoder(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs,
                 param_dtype=jnp.float32, use_bias=False):
        self.kernel = NonnegativeParam(nnx.nn.linear.default_kernel_init(
            rngs.params(), (in_features, out_features), param_dtype
        ))
        if use_bias:
            self.bias = nnx.Param(nnx.nn.linear.default_bias_init(
                rngs.params(), (out_features,), param_dtype
            ))

    def __call__(self, inputs):
        kernel = self.kernel[...]
        hs = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)),
                                              ((), ())),
                             out_sharding=None, precision=None)
        if hasattr(self, "bias"):
            bias = self.bias[...]
            hs += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return hs

class PVaePrior(nnx.Module):
    def __init__(self, z_dim, *, rngs: nnx.Rngs):
        self.log_rate = nnx.Param(rngs.uniform(shape=(z_dim,), minval=-6.,
                                               maxval=-4.))

    def __call__(self, natural=False, rngs=None):
        if natural:
            return self.log_rate.value
        return jnp.exp(self.log_rate)

def pvae_guide(xs, encoder: PVaeEncoder):
    encoder = nnx_module("encoder", encoder)
    u = encoder(xs)
    with numpyro.plate("batch", xs.shape[0]):
        return numpyro.sample("z", dist.Poisson(jnp.exp(u)).to_event(1))

def pvae_model(xs, decoder: NMFDecoder, prior: PVaePrior, scale=None, **kwargs):
    decoder = nnx_module("decoder", decoder)
    prior = nnx_module("prior", prior)
    if scale is None:
        scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", xs.shape[0]):
        z = numpyro.sample("z", dist.Poisson(prior()).to_event(1))
        loc = decoder(z).reshape(xs.shape)
        return numpyro.sample("x", dist.Normal(loc, scale).to_event(3), obs=xs)

def main(args):
    guide_rng, model_rng, rng_key = random.split(key(0), 3)
    train_init, train_fetch = load_dataset(
        MNIST, batch_size=args.batch_size, split="train"
    )
    test_init, test_fetch = load_dataset(
        MNIST, batch_size=args.batch_size, split="test"
    )
    num_train, train_idx = train_init()
    rng_key, rng_key_init = random.split(rng_key, 2)
    sample_batch = train_fetch(0, train_idx)[0].reshape(-1, 1, 28, 28)

    encoder = PVaeEncoder(args.z_dim, rngs=nnx.Rngs(guide_rng), x_dim=28)
    guide = functools.partial(pvae_guide, encoder=encoder)
    l_guide = dsgd.count_layers(guide, sample_batch)



    dec_rng, prior_rng = random.split(model_rng)
    decoder = NMFDecoder(args.z_dim, 28 * 28, rngs=nnx.Rngs(dec_rng))
    prior = PVaePrior(args.z_dim, rngs=nnx.Rngs(prior_rng))
    model = functools.partial(pvae_model, decoder=decoder, prior=prior)
    l_model = dsgd.count_layers(model, sample_batch)

    l = max(l_guide, l_model)
    schedule = dsgd.eta_schedule(K=args.num_epochs, ell=l, eta_final=0.08)
    guide = dsgd.dsgd(guide)
    model = dsgd.dsgd(model)


    adam = optim.Adam(args.learning_rate)
    svi = SVI(model, guide, adam, Trace_ELBO())

    svi_state = svi.init(rng_key_init, schedule[0], sample_batch)

    # A single jitted optimisation step. We drive the per-epoch batch loop from
    # Python (rather than wrapping the whole epoch in one lax.fori_loop) so that
    # tqdm reports live per-batch progress: with the DSGD adaptive sampler the
    # per-step cost grows with the Poisson rate exp(u), so a live signal makes it
    # obvious when training starts to diverge and slow down. eta is passed as an
    # argument (same shape/dtype every step) so the step compiles only once.
    @jit
    def train_step(eta, svi_state, batch):
        return svi.update(svi_state, eta, batch)

    @jit
    def eval_test(eta, svi_state, rng_key, test_idx):
        def body_fun(i, loss_sum):
            batch = test_fetch(i, test_idx)[0].reshape(-1, 1, 28, 28)
            # FIXME: does this lead to a requirement for an rng_key arg in svi_eval?
            loss = svi.evaluate(svi_state, eta, batch) / len(batch)
            loss_sum += loss
            return loss_sum

        loss = lax.fori_loop(0, num_test, body_fun, 0.0)
        loss = loss / num_test
        return loss

    def reconstruct_img(epoch, rng_key):
        img = test_fetch(0, test_idx)[0][0].reshape(1, 1, 28, 28)
        plt.imsave(
            os.path.join(RESULTS_DIR, "original_epoch={}.png".format(epoch)),
            img.squeeze(),
            cmap="gray",
        )
        test_sample = img
        params = svi.get_params(svi_state)

        from numpyro.infer.util import get_importance_trace

        with numpyro.handlers.seed(rng_seed=rng_key):
            model_trace, guide_trace = get_importance_trace(model, guide,
                                                            (schedule[epoch],
                                                             test_sample),
                                                            {}, params)
        img_loc = model_trace["x"]["fn"].mean.squeeze()
        plt.imsave(
            os.path.join(RESULTS_DIR, "recons_epoch={}.png".format(epoch)),
            img_loc,
            cmap="gray",
        )

    for i in range(args.num_epochs):
        rng_key, rng_key_test, rng_key_reconstruct = random.split(rng_key, 3)
        t_start = time.time()
        num_train, train_idx = train_init()
        epoch_loss = 0.0
        progress = tqdm(
            range(num_train), desc="epoch {}".format(i), unit="batch"
        )
        for j in progress:
            batch = train_fetch(j, train_idx)[0].reshape(-1, 1, 28, 28)
            svi_state, loss = train_step(schedule[i], svi_state, batch)
            epoch_loss += loss
            progress.set_postfix(loss="{:.1f}".format(float(loss) / len(batch)))
        num_test, test_idx = test_init()
        test_loss = eval_test(schedule[i], svi_state, rng_key_test, test_idx)
        reconstruct_img(i, rng_key_reconstruct)
        print(
            "Epoch {}: test loss = {} ({:.2f} s.)".format(
                i, test_loss, time.time() - t_start
            )
        )


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.20.0")
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=15, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
    )
    parser.add_argument("-batch-size", default=128, type=int, help="batch size")
    parser.add_argument("-z-dim", default=1568, type=int, help="size of latent")
    args = parser.parse_args()
    main(args)
