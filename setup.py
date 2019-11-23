import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-trainer",
    version="0.8.1",
    description="Lightweight wrapper around PyTorch ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theSoenke/pytorch-trainer",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.2.0',
        'tqdm>=4.35.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
