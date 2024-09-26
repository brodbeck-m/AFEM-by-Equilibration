#!/bin/bash
set -e
CONTAINER_ENGINE="docker"

if [ "$AFEM_HOME" = "" ];
then
    # Error if path to dolfinx_eqlb i not set
    echo "Patch to source folder not set! Use "export AFEM_HOME=/home/.../AFEM-by-Equilibration""
    exit 1
else
    # Build docker image
    echo "DEQLB_HOME is set to '$AFEM_HOME'"
    ${CONTAINER_ENGINE} pull dolfinx/dolfinx:v0.6.0-r1
    ${CONTAINER_ENGINE} build --no-cache -f "${AFEM_HOME}/docker/Dockerfile" -t brodbeck-m/afem-by-eqlb:release .
fi
