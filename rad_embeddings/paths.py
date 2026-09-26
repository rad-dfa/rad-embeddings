import os
import re

CHECKPOINT_PREFIX = "encoder_params"
CHECKPOINT_EXT = ".msgpack"
LOG_PREFIX = "log"
LOG_EXT = ".csv"

# The sampler and p keys are optional only to read names written before they existed; run_name always writes them.
_RUN_PATTERN = r"max_size_(\d+)_n_tokens_(\d+)_seed_(\d+)_binary_reward_(True|False)_gamma_([\d\.eE+-]+)(?:_sampler_([A-Za-z]+)_p_(None|[\d\.eE+-]+))?"
# Every run before those keys existed was trained with RADSampler and its default p (0.5 in every dfax release).
_LEGACY_SAMPLER = "RAD"
_LEGACY_P = "0.5"
_CHECKPOINT_RE = re.compile(rf"{CHECKPOINT_PREFIX}_{_RUN_PATTERN}{re.escape(CHECKPOINT_EXT)}")
_LOG_RE = re.compile(rf"{LOG_PREFIX}_{_RUN_PATTERN}{re.escape(LOG_EXT)}")


def default_storage_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "storage")


def run_name(max_size: int, n_tokens: int, seed: int, binary_reward: bool, gamma: float, sampler: str, p: float | None) -> str:
    # repr(float(...)) round-trips exactly, so 0.9 from the CLI and 0.9 from Python give the same name
    p = "None" if p is None else repr(float(p))
    return f"max_size_{int(max_size)}_n_tokens_{int(n_tokens)}_seed_{int(seed)}_binary_reward_{bool(binary_reward)}_gamma_{float(gamma)!r}_sampler_{sampler}_p_{p}"


def checkpoint_path(save_dir: str, **run) -> str:
    return os.path.join(save_dir, f"{CHECKPOINT_PREFIX}_{run_name(**run)}{CHECKPOINT_EXT}")


def log_path(save_dir: str, **run) -> str:
    return os.path.join(save_dir, f"{LOG_PREFIX}_{run_name(**run)}{LOG_EXT}")


def _parse(regex: re.Pattern, fname: str) -> dict | None:
    m = regex.fullmatch(os.path.basename(fname))
    if m is None:
        return None
    max_size, n_tokens, seed, binary_reward, gamma, sampler, p = m.groups()
    p = p or _LEGACY_P
    try:
        gamma = float(gamma)
        p = None if p == "None" else float(p)
    except ValueError:
        return None
    return {
        "max_size": int(max_size),
        "n_tokens": int(n_tokens),
        "seed": int(seed),
        "binary_reward": binary_reward == "True",
        "gamma": gamma,
        "sampler": sampler or _LEGACY_SAMPLER,
        "p": p,
    }


def parse_checkpoint_name(fname: str) -> dict | None:
    return _parse(_CHECKPOINT_RE, fname)


def parse_log_name(fname: str) -> dict | None:
    return _parse(_LOG_RE, fname)
