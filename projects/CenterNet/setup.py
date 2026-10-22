import glob
import os
import shutil

from setuptools import find_packages
from setuptools import setup


def get_version():
    init_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "centernet", "__init__.py")
    init_py = open(init_py_path).readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


def get_model_zoo_configs() -> list[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    centernet/model_zoo.
    """
    # Use absolute paths while symlinking.
    source_configs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    destination = os.path.join(os.path.dirname(os.path.realpath(__file__)), "centernet", "model_zoo", "configs")
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if os.path.exists(source_configs_dir):
        if os.path.islink(destination):
            os.unlink(destination)
        elif os.path.isdir(destination):
            shutil.rmtree(destination)

    if not os.path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.py", recursive=True)
    return config_paths


setup(
    name="centernet",
    version=get_version(),
    author="tkhe",
    url="https://github.com/tkhe/OverDrive/tree/master/projects/CenterNet",
    description="CenterNet object detection example built on OverDrive.",
    packages=find_packages(exclude=("configs", "tests*")),
    package_data={"centernet.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.10",
    install_requires=["overdrive>=1.0.0"],
)
