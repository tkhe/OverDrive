import glob
import os
import shutil

from setuptools import find_packages
from setuptools import setup


def get_version():
    init_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dn_detr", "__init__.py")
    with open(init_py_path) as file:
        init_py = file.readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


def get_model_zoo_configs() -> list[str]:
    """
    Include the project's configuration files in the model zoo package.
    """
    project_dir = os.path.dirname(os.path.realpath(__file__))
    source_configs_dir = os.path.join(project_dir, "configs")
    destination = os.path.join(project_dir, "dn_detr", "model_zoo", "configs")

    if os.path.islink(destination):
        os.unlink(destination)
    elif os.path.isdir(destination):
        shutil.rmtree(destination)

    if os.path.isdir(source_configs_dir):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            shutil.copytree(source_configs_dir, destination)

    return glob.glob("configs/**/*.py", recursive=True)


setup(
    name="dn-detr",
    version=get_version(),
    author="tkhe",
    url="https://github.com/tkhe/OverDrive/tree/master/projects/DN-DETR",
    description="DN-DETR object detection example built on OverDrive.",
    packages=find_packages(exclude=("configs", "tests*")),
    package_data={"dn_detr.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.10",
    install_requires=["overdrive>=1.0.0"],
)
