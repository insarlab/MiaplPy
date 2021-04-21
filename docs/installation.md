## Install MiNoPy

1. Set the following environment variables in your source file. It could be ~/.bash_profile file for bash user or ~/.cshrc file for csh/tcsh user.

   ##### MinSAR is optional for those who use HPC job submission 

```
if [ -z ${PYTHONPATH+x} ]; then export PYTHONPATH=""; fi

##--------- MintPy ------------------##
export MINTPY_HOME=~/tools/MintPy
export PYTHONPATH=${PYTHONPATH}:${MINTPY_HOME}
export PATH=${PATH}:${MINTPY_HOME}/mintpy

##--------- PyAPS -------------------##
export PYAPS_HOME=~/tools/PyAPS
export PYTHONPATH=${PYTHONPATH}:${PYAPS_HOME}

##--------- MinSAR ------------------##
export MINSAR_HOME=~/tools/rsmas_insar
export PYTHONPATH=${PYTHONPATH}:${MINSAR_HOME}
export PATH=${PATH}:${MINSAR_HOME}/minsar:${MINSAR_HOME}/minsar/utils

#---------- MiNoPy ------------------##
export MINOPY_HOME=~/tools/MiNoPy
export PYTHONPATH=${PYTHONPATH}:${MINOPY_HOME}
export PATH=${PATH}:${MINOPY_HOME}/minopy

```

2. Install [MintPy, PyAPS](https://github.com/insarlab/MintPy/blob/main/docs/installation.md) and pre-requisites\

3. Install [ISCE](https://github.com/isce-framework/isce2) and cython\
Use the [guide](https://github.com/isce-framework/isce2) or install with conda:\
`conda install -c conda-forge isce2 `\
`conda install cython`

4. Install [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/)

5. Clone the repo [MinSAR](https://github.com/geodesymiami/rsmas_insar) (optional)\
`git clone https://github.com/geodesymiami/rsmas_insar.git $MINSAR_HOME`

6. Clone the repo [MiNoPy](https://github.com/geodesymiami/MiNoPy)\
`git clone https://github.com/geodesymiami/MiNoPy.git $MINOPY_HOME`\
`cd $MINOPY_HOME; ./install`
