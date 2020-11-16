# Replay Overshooting: Learning Stochastic Latent Dynamics with the Extended Kalman Filter

This package contains code to use the methods described in our submission to the International
Conference on Robotics and Automation (ICRA) 2021:

<blockquote>
    Albert H. Li*, Philipp Wu*, Monroe Kennedy III
    <strong>
         Replay Overshooting: Learning Stochastic Latent
         Dynamics with the Extended Kalman Filter
    </strong>
</blockquote>

### Minimal Setup
Note that only `python>=3.8` is supported.

```
git clone https://github.com/wuphilipp/replay-overshooting.git
cd replay-overshooting
pip install -e .
```

We recommend using [conda](https://docs.conda.io/en/latest/) for managing python
environments. Optionally, install the requirements.txt for development utilities.
```
pip install -r requirements.txt
```

Run the example
```
python scripts/train_example.py
```

---

### Repository Overview

#### File Structure
 * `scripts` - Scripts used for training and evaluation.
   * `paper_experiments`
     * `pend_img` - Configuration files for training models on video frames of a
     simulated pendulum.
     * `mit_push` - Configuration files for training models on the [MIT Push Dataset](https://mcube.mit.edu/push-dataset/index.html).
 * `dynamics_learning` - Contains the core code.
   * `custom` - Common learning rate schedulers and policies.
   * `data` - Manages data and converts datasets into a common format.
   * `networks` - Contains all code for creating neural dynamics models.
     * `baseline` - Includes implementations of baseline models from
     [PlaNet](https://planetrl.github.io/).
     * `image_models` - Implements common vision models and extends the `EKF` to
     image observations.
     * `kalman` - Contains all core functional for the `EKF`.
   * `traininig` - Manages training and evaluation of models.
   * `utils` - Common utilities shared across the code base.

More detailed documentation is provided in the code.

#### Training and Evaluation
All runs are managed through an `ExpConfig` (found in
`dynamics_learning/training/configs.py`) which contains all the information
necessary to reproduce a model (including hyperparameter settings). Each
experiment in the `scripts` directory contains the
construction of the `ExpConfig`. See `scripts/train_example.py` for an example
of this. Model performance evaluation can done by running the corresponding
`eval_*` file in the same folder.

Tracking model metrics and visualizations can be viewed through
[Tensorboard](https://www.tensorflow.org/tensorboard). Tensorboard logs will
automatically be created during training in `log` folder.


### Additional Code

- **[pytorch](https://pytorch.org)** is auto differentiation framework used
  throughout the codebase.
- **[fannypack](https://github.com/brentyi/fannypack)** is used for experiment
  management.

---

### Contribute

Contributions and bug fixes are welcome! The code follows the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
style guide for docstrings. In addition, the following tools are used to manage
the python code:

  - **[black](https://github.com/psf/black)** for code formatting
  - **[isort](https://github.com/pycqa/isort/)** for managing imports
  - **[flake8](https://flake8.pycqa.org/en/latest/)** for style checking
  - **[mypy](http://mypy-lang.org/)** for static type checking


#### pre-commit hooks

Some pre-commits are provided in the repo to be optionally used to assist development. This will automatically do some checking/formatting. To use the pre-commit hooks, run the following:

```
pip install pre-commit
pre-commit install
```

---

### Known Bugs
This code base makes heavy use of `torch.cholesky`. However occasionally there
will be a CUDA `illegal memory access` error. This is a known [pytorch issue](https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624),
but not something we can directly resolve. If this occurs, try changing the
random seed.

