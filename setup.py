from setuptools import setup, find_packages

setup(
    name='nam_pytorch',
    version='0.0.1',
    description='Neural Additive Models (NAM) implemented in PyTorch',
    author='Zheng Xin',
    author_email='zhengncst@gmail.com',  
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'scikit-learn',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
)