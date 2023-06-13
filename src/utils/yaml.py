from pathlib import Path

from ruamel.yaml import YAML


def load_yaml_files(files):
    merged_data = {}
    yaml = YAML()
    for file in files:
        with open(file, "r") as f:
            data = yaml.load(f)
            merged_data.update(data)
    return merged_data


def load_yaml_file(file):
    yaml = YAML()
    with open(file, "r") as f:
        data = yaml.load(f)
    return data


def load_config_from_dir(dir: Path):
    files = dir.glob("**/*.yml")
    return load_yaml_files(files)


__all__ = [
    "load_yaml_files",
    "load_yaml_file",
    "load_config_from_dir",
]
