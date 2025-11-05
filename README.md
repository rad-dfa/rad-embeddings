![Rad Embeddings Logo](https://rad-embeddings.github.io/assets/logo.svg)

This repo contains a JAX implementation of RAD embeddings, see [project webpage](https://rad-embeddings.github.io/) for more information.

# Installation

This package will soon be made pip-installable and replace [this version](https://github.com/RAD-Embeddings/rad-embeddings). In the meantime, pull the repo and and install locally.

```
git clone https://github.com/rad-dfa/rad-embeddings.git
pip install -e rad-embeddings
```

# Usage

Instantiate pretrained encoder.

```python
encoder = Encoder() # Creates a DFA encoder with default parameters: handles at most 10-state DFAs with 10-token alphabets
```

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