import jax
import jax.numpy as jnp
from dfax import DFAx
from dfax.samplers import RADSampler, ReachSampler
from rad_embeddings import Encoder

def generate_chain_dfax(n_states, n_tokens=10):
    sampler = RADSampler(max_size=n_states, n_tokens=n_tokens)
    dfax = sampler.trivial(False)
    start = dfax.start
    transitions = dfax.transitions
    labels = dfax.labels.at[-1].set(True)
    for i in range(n_states - 1):
        transitions = transitions.at[i % n_states, i % n_tokens].set(i + 1)
    return DFAx.create(
        start=0,
        transitions=transitions,
        labels=labels
    )


key = jax.random.PRNGKey(42)
encoder = Encoder(max_size=None) # Unbounded message passing, enabled with version 0.2.4 of rad_embeddings

n_states = 100
n_tokens = 10

sampler = ReachSampler(min_size=n_states, max_size=n_states, n_tokens=n_tokens)

key, subkey = jax.random.split(key)
some_dfax = sampler.sample(subkey)
some_rad = encoder(some_dfax)

dfax = generate_chain_dfax(n_states, n_tokens=n_tokens)
rad = encoder(dfax)
d_baseline = encoder.distance(rad, rad)

assert dfax != some_dfax
some_d = encoder.distance(rad, some_rad)
# If some_d - d_baseline < 1e-8 is False, then the messages haven't converged yet.
print("Distance to some random Reach DFAx:", some_d, some_d - d_baseline < 1e-8)

for i in range(n_states - 1):
    key, subkey = jax.random.split(key)
    offset = jax.random.randint(subkey, shape=(1,), minval=1, maxval=n_tokens).item()
    dfax_p = DFAx.create(
        start=dfax.start,
        transitions=dfax.transitions.at[i % n_states, i % n_tokens].set(i).at[i % n_states, (i + offset) % n_tokens].set(i + 1),
        labels=dfax.labels
    )
    assert dfax != dfax_p
    rad_p = encoder(dfax_p)
    d = encoder.distance(rad, rad_p)
    print(i, d, d - d_baseline < 1e-8)

# An output:
# Distance to some random Reach DFAx: [[0.05994296]]
# 0 [[0.07843833]] [[False]]
# 1 [[0.00976952]] [[False]]
# 2 [[0.00291922]] [[False]]
# 3 [[0.00035061]] [[False]]
# 4 [[0.00034647]] [[False]]
# 5 [[0.00034539]] [[False]]
# 6 [[0.00034527]] [[ True]]
# 7 [[0.00034527]] [[ True]]
# 8 [[0.00034527]] [[ True]]
# 9 [[0.00034527]] [[ True]]
# ...
# This is a representative pattern observed across different runs.

