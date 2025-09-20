## Contributing to Audio2Face-3D Training Framework

### Contribution Policy

This repository is maintained by NVIDIA's Audio2Face team. We welcome contributions to this repository including:

- **Bug reports** with detailed reproduction steps
- **Pull requests** for bug fixes, improvements, and new features
- **Documentation improvements** and examples
- **Community support** helping other users with questions and issues

All contributions should follow the guidelines below and include proper testing.

### Development Setup

#### Prerequisites
- Linux/WSL2, CUDA GPU (6GB+ VRAM), Docker, NGC access
- See [README.md](README.md#prerequisites) for details

#### Environment Setup
```bash
cp .env.example .env  # Edit with your paths
chmod +x docker/*.sh
./docker/build_docker.sh
```

#### VSCode Development (Recommended)
The project includes pre-configured VSCode settings for:
- Black formatting (120 characters)
- Debug configurations for all framework stages
- Dev container support

For detailed instructions, see [VSCode Development and Debugging](docs/training_framework.md#vscode-development-and-debugging--advanced) in the Training Framework documentation.

### Code Standards

#### Before Committing
**Always run before committing:**
```bash
python tools/format_code.py
```

This handles Black formatting (120 characters) and ensures consistency. VSCode will partially auto-format during development, but this tool ensures final consistency.

#### Required for Any Contributions
- Python type hints
- License headers (see existing files for format)
- Test with example dataset
- Docker compatibility

### Developer Certificate of Origin (DCO)

All contributions must comply with the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

#### Signing Your Commits

Sign off on all commits using the `--signoff` (or `-s`) option:

```bash
git commit -s -m "Add feature X"
```

This appends a sign-off line to your commit message:
```
Signed-off-by: Your Name <your@email.com>
```

#### DCO Requirements

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### License

This project is licensed under the [Apache License 2.0](LICENSE). All contributions must be compatible with this license.
