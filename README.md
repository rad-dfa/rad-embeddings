![Rad Embeddings Logo](https://rad-embeddings.github.io/assets/logo.svg)

This repo contains a JAX implementation of RAD embeddings, see [project webpage](https://rad-embeddings.github.io/) for more information.

# Installation

Install using pip.

```
pip install rad-embeddings
```

# Usage

Instantiate a pretrained encoder.

```python
from rad_embeddings import Encoder

encoder = Encoder() # Loads a pretrained DFA encoder with default parameters: handles at most 10-state DFAs with 10-token alphabets
```

Use [DFAx](https://github.com/rad-dfa/dfax) to sample DFAs.

```python
import jax
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler

sampler = RADSampler()
key = jax.random.PRNGKey(42)

key, subkey = jax.random.split(key)
dfax = sampler.sample(subkey)
```

Pass the DFA to the encoder to get its embedding &mdash; both [DFAx](https://github.com/rad-dfa/dfax) and [DFA](https://github.com/mvcisback/dfa) objects are supported.

```python
from dfax import dfax2dfa

rad_embed_from_dfax = encoder(dfax)
rad_embed_from_dfa = encoder(dfax2dfa(dfax)) # Returns the same embedding
```

Compute bisimulation distance between two DFA embeddings.

```python
key, subkey = jax.random.split(key)
dfax_l = sampler.sample(subkey)
rad_l = encoder(dfax_l)

key, subkey = jax.random.split(key)
dfax_r = sampler.sample(subkey)
rad_r = encoder(dfax_r)

distance = encoder.distance(rad_l, rad_r)
```

Solve a one-step bisimulation problem.

```python
from dfax import DFAx
import jax.numpy as jnp

# Reach token 1 and then token 2 while avoding token 9
dfa_l = DFAx.create(
	start = 0,
	transitions = jnp.array([
		[0, 1, 0, 0, 0, 0, 0, 0, 0, 3],
		[1, 1, 2, 1, 1, 1, 1, 1, 1, 3],
		[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
		[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
		[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
	]),
	labels = jnp.array([False, False, True, False, False])
)

# Reach token 9
dfa_r = DFAx.create(
	start = 0,
	transitions = jnp.array([
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
		[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
		[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
	]),
	labels = jnp.array([False, True, False, False, False])
)

distinguishing_action = encoder.solve(dfa_l, dfa_r) # Returns 1 as token 1 is the one-step distinguishing action
```

<!--To train your own encoder, first clone the repo and then use [train.py](https://github.com/rad-dfa/rad-embeddings/blob/main/rad_embeddings/train.py).

```bash
git clone https://github.com/rad-dfa/rad-embeddings.git

uv run train.py --seed 42 --save-dir storage --max-size 10 --n-tokens 10 --debug
```-->

See [train](https://github.com/rad-dfa/rad-embeddings/blob/main/rad_embeddings/train.py) and [test](https://github.com/rad-dfa/rad-embeddings/blob/main/test.py) scripts for more.



# Citation

Please cite the following papers if you use RAD Embeddings in your work.

```
@inproceedings{DBLP:conf/nips/YalcinkayaLVS24,
  author       = {Beyazit Yalcinkaya and
                  Niklas Lauffer and
                  Marcell Vazquez{-}Chanlatte and
                  Sanjit A. Seshia},
  title        = {Compositional Automata Embeddings for Goal-Conditioned Reinforcement
                  Learning},
  booktitle    = {NeurIPS},
  year         = {2024}
}
```

```
@inproceedings{DBLP:conf/neus/YalcinkayaLVS25,
  author       = {Beyazit Yalcinkaya and
                  Niklas Lauffer and
                  Marcell Vazquez{-}Chanlatte and
                  Sanjit A. Seshia},
  title        = {Provably Correct Automata Embeddings for Optimal Automata-Conditioned
                  Reinforcement Learning},
  booktitle    = {NeuS},
  series       = {Proceedings of Machine Learning Research},
  volume       = {288},
  pages        = {661--675},
  publisher    = {{PMLR}},
  year         = {2025}
}
```