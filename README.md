# 
# <a name="AFEMbyEqlb"></a> Adaptive finite element methods based on flux equilibration using FEniCSx
[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](https://www.repostatus.org/badges/latest/inactive.svg)](https://www.repostatus.org/#inactive) [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--4479-d45815.svg)](https://doi.org/10.18419/darus-4479)

Author: Maximilian Brodbeck

This repository showcases how adaptive finite element solvers using equilibration based a posteriori error estimates can be build. Therefore, FEniCSx [[1]](#1) alongside with [dolfinx_eqlb](https://github.com/brodbeck-m/dolfinx_eqlb) [[2]](#2), an extension for efficient flux equilibration are used. Classical benchmarks for the Poisson problem (L-shaped domain [[1]](#1) and variants of the Kellogg problem [[3]](#3)[[4]](#4)) and linear elasticity (manufactured solution and Cooks membrane [[5]](#5)) are shown. The here presented code can be used to reproduce the results in the [related publication](???).

* [Features](#features)
* [Getting started](#getting-started)
* [How to cite](#how-to-cite)
* [Literature](#literature)
* [License](#license)

# <a id="features"></a> Features
AFEM-by-Equilibration contains adaptive finite element algorithms based on equilibrated fluxes for
- the Poisson problem on an L-Shaped domain
- the Poisson problem on a squared domain with discontinuous coefficients
- linear elasticity on a unit-square with manufactured solution
- linear elasticity on the Cooks membrane

# <a id="getting-started"></a> Getting started
The codes can be used based on a docker container. Therefore:

1. Clone this repository using the command:

```shell
git clone https://github.com/brodbeck-m/AFEM-by-Equilibration.git
```

2. Download the required Docker image of DOLFINx:

```shell
docker pull dolfinx/dolfinx:v0.6.0-r1
```

3. Build a Docker image containing dolfinx_eqlb

```shell
./docker/build_image.sh 
```

4. Launch the docker container and run ```execute_all.sh``` in order to reproduce all results

```shell
./docker/launch-container.sh

# Execute all calculations
./execute_all.sh
```

Alternatively, a ready-to-use docker image is provided via [DaRUS](Add Link!).

# <a id="how-to-cite"></a> How to cite
AFEM-by-Equilibration was cerated for research proposes. The code can be cited via [DaRUS](https://doi.org/10.18419/darus-4479).

If you are using using AFEM-by-Equilibration please cite

**Adaptive finite element methods based on flux equilibration using FEniCSx**. preprint (2024). doi: [???](https://doi.org/10.18419/darus-4498)
```bib
@article{Brodbeck2024,
    doi = {???},
    year = {2024},
    publisher = {},
    author = {M. Brodbeck, F. Bertrand and T. Ricken},
    title = {Adaptive finite element methods based on flux equilibration using FEniCSx},
    journal = {}}
```



# <a id="literature"></a> Literature
<a id="1">[1]</a> Baratta, I. A. et al.: DOLFINx: The next generation FEniCS problem solving environment. preprint. (2023) doi: [10.5281/zenodo.10447666](https://doi.org/10.5281/zenodo.10447666)

<a id="2">[2]</a> Brodbeck, M., Bertrand, F. and Ricken, T.: dolfinx_eqlb v1.2.0. DaRUS (2024) doi: [10.18419/darus-4498](https://doi.org/10.18419/darus-4498)

<a id="3">[3]</a> Kellogg, R.B.: On the poisson equation with intersecting interfaces. Appl. Anal. (1974) doi: [10.1080/00036817408839086](https://doi.org/10.1080/00036817408839086)

<a id="4">[4]</a> Rivière, B. and Wheeler, M.F.: A Posteriori error estimates for a discontinuous galerkin method applied to elliptic problems. Comput. Math. Appl. (2003) doi: [10.1016/S0898-1221(03)90086-1](https://doi.org/10.1016/S0898-1221(03)90086-1)

<a id="5">[5]</a> Schröder, J. et al.: A Selection of Benchmark Problems in Solid Mechanics and Applied Mathematics. Arch. Comput. Methods Eng. (2020) doi: [0.1007/s11831-020-09477-3](https://doi.org/10.1007/s11831-020-09477-3)

<a id="6">[6]</a> Strang, G. and Fix, G.: An Analysis of the Finite Element Methods. Wellesley-Cambridge Press, Philadelphia (2008) 

# <a id="license"></a> License
AFEM-by-Equilibration is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AFEM-by-Equilibration is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with AFEM-by-Equilibration. 
If not, see <https://www.gnu.org/licenses/>.
