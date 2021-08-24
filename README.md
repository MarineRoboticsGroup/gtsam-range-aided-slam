# qcqp-range-only-slam

Formulates the range-only SLAM problem as a QCQP and solves it as such using
off-the-shelf solvers

## Getting Started

This library is quite a pain to get the dependencies right, so please just
accept the ugliness of pulling the `QCQP` source code directly into this repo
and use `pip` to create a working environment from the `requirements.txt`
provided in this repo. There are definitely some unnecessary installs in there
that could used to be cleaned up, but lets deal with it for now. My way of doing
this is (ugly) but as follows:

```bash
conda create --name qcqp_ro_slam python=3.9
pip install -r requirements.txt
```
