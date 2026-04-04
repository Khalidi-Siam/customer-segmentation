from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements from a file and returns a list of requirements.
    '''
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='customer_segmentation',
    version='0.0.1',
    author='Khalidi Siam',
    author_email="siam074@yahoo.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)