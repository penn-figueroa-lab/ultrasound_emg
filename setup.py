from setuptools import setup
from distutils.core import setup

package_name = 'human_model'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=[
        'setuptools>=60.0.0',
        'absl-py',
        'lxml',
        "mujoco",
        'numpy',
        'pyyaml',
        'torch',
        'matplotlib',
        'pytorch_seed',
        'pytorch_kinematics', # @ git+https://github.com/UM-ARM-Lab/pytorch_kinematics.git'
    ],
    # dependency_links=[
    #     'git+https://github.com/UM-ARM-Lab/pytorch_kinematics.git#egg=pytorch_kinematics'

    # ],
)