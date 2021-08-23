# qcqp-range-only-slam

Formulates the range-only SLAM problem as a QCQP and solves it as such using
off-the-shelf solvers

## Getting Started

This library makes use of the nonconvex QCQP solver developed by Jaehyun Park
and Stephen Boyd. As such we have to be careful with the versions of packages we
use. In particular, we must use `python[version='2.8.*|3.5.*|3.6.*']` and
`cvxpy[version=0.4.9]`.

To set up a proper environment using `conda` you can execute the following:

```bash
conda create --name qcqp_ro_slam python=3.6
conda activate qcqp_ro_slam
conda install -c cvxgrp cvxpy=0.4.9
```
