# Security policy

## Supported versions

The latest release on PyPI. This project is a research-grade plugin; older versions receive no
backports.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Lfan-ke/ES-MoE/security/advisories/new), or
mail the maintainer if you cannot. Please do not open a public issue for anything exploitable.

## Scope notes

`esmoe` loads model configurations and patches the task model of an already-installed `ultralytics`.
It executes no shell commands and downloads nothing. Loading a model config or a checkpoint from an
untrusted source is as dangerous here as it is anywhere else in the PyTorch ecosystem, and that risk
belongs to the loader, not to this package.
