import gc
import random

import dotenv
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from src import utils


@pytest.fixture(autouse=True)
def deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the random seed for deterministic results.

    Args:
    ----
        monkeypatch (fixture): Pytest fixture for monkey-patching.

    """
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def gc_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Perform garbage collection on the GPU after each test.

    Args:
    ----
        monkeypatch (fixture): Pytest fixture for monkey-patching.

    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the same optional utilities used in the entrypoint scripts.

    Args:
    ----
        monkeypatch (fixture): Pytest fixture for monkey-patching.

    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="main.yaml", return_hydra_config=True, overrides=[])

        # disable unnecessary extras
        with open_dict(cfg):
            cfg.extras.ignore_warnings = False
            cfg.extras.print_config = False

        utils.extras(cfg)


@pytest.fixture(autouse=True)
def load_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the environment variables from the .env file.

    Args:
    ----
        monkeypatch (fixture): Pytest fixture for monkey-patching.

    """
    dotenv.load_dotenv()
