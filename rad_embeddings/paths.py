import os
import re

CHECKPOINT_PREFIX = "encoder_params"
CHECKPOINT_EXT = ".msgpack"
LOG_PREFIX = "log"
LOG_EXT = ".csv"

_RUN_PATTERN = r"max_size_(\d+)_n_tokens_(\d+)_seed_(\d+)_binary_reward_(True|False)_gamma_([\d\.eE+-]+)(_experimental)?"
_CHECKPOINT_RE = re.compile(rf"{CHECKPOINT_PREFIX}_{_RUN_PATTERN}{re.escape(CHECKPOINT_EXT)}")
_LOG_RE = re.compile(rf"{LOG_PREFIX}_{_RUN_PATTERN}{re.escape(LOG_EXT)}")


def default_storage_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "storage")


def run_name(max_size: int, n_tokens: int, seed: int, binary_reward: bool, gamma: float, experimental: bool = False) -> str:
    # repr(float(...)) round-trips exactly, so 0.9 from the CLI and 0.9 from Python give the same name
    name = f"max_size_{int(max_size)}_n_tokens_{int(n_tokens)}_seed_{int(seed)}_binary_reward_{bool(binary_reward)}_gamma_{float(gamma)!r}"
    # Suffix only when set, so names of standard runs are unaffected
    return f"{name}_experimental" if experimental else name


def checkpoint_path(save_dir: str, **run) -> str:
    return os.path.join(save_dir, f"{CHECKPOINT_PREFIX}_{run_name(**run)}{CHECKPOINT_EXT}")


def log_path(save_dir: str, **run) -> str:
    return os.path.join(save_dir, f"{LOG_PREFIX}_{run_name(**run)}{LOG_EXT}")


def _parse(regex: re.Pattern, fname: str) -> dict | None:
    m = regex.fullmatch(os.path.basename(fname))
    if m is None:
        return None
    max_size, n_tokens, seed, binary_reward, gamma, experimental = m.groups()
    try:
        gamma = float(gamma)
    except ValueError:
        return None
    return {
        "max_size": int(max_size),
        "n_tokens": int(n_tokens),
        "seed": int(seed),
        "binary_reward": binary_reward == "True",
        "gamma": gamma,
        "experimental": experimental is not None,
    }


def parse_checkpoint_name(fname: str) -> dict | None:
    return _parse(_CHECKPOINT_RE, fname)


def parse_log_name(fname: str) -> dict | None:
    return _parse(_LOG_RE, fname)
