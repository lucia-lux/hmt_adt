from pathlib import Path
from strictyaml import YAML, load

# Project Directories
PACKAGE_ROOT = Path(__file__).parent.resolve()
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = ROOT / "config.yml"
DATASET_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "trained_models"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


config = fetch_config_from_yaml()