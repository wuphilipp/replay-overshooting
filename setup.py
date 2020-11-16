import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="replay-overshooting",
    version="0.0.1",
    author="Albert Li, Philipp Wu",
    author_email="ahli@stanford.edu, philippwu@berkeley.edu",
    description="Package for method described in `Replay Overshooting: Learning Stochastic Latent Dynamics with the Extended Kalman Filter`.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wuphilipp/replay-overshooting",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "fannypack",
        "numpy",
        "matplotlib",
        "sdeint>=0.2.1",
        "tensorboard>=2.2.0",
        "torch>=1.6.0",
        "torchdiffeq>=0.1.1",
    ],
)
