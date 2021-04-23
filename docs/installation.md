## Install MiNoPy

#### 1. Set the following environment variables in your source file. It could be ~/.bash_profile file for bash user or ~/.cshrc file for csh/tcsh user.

```
if [ -z ${PYTHONPATH+x} ]; then export PYTHONPATH=""; fi

##--------- MintPy ------------------##
export MINTPY_HOME=~/tools/MintPy
export PYTHONPATH=${PYTHONPATH}:${MINTPY_HOME}
export PATH=${PATH}:${MINTPY_HOME}/mintpy

##--------- PyAPS -------------------##
export PYAPS_HOME=~/tools/PyAPS
export PYTHONPATH=${PYTHONPATH}:${PYAPS_HOME}

#---------- MiNoPy ------------------##
export MINOPY_HOME=~/tools/MiNoPy
export PYTHONPATH=${PYTHONPATH}:${MINOPY_HOME}
export PATH=${PATH}:${MINOPY_HOME}/minopy

#---------- snaphu (optional) -------##
export SNAPHU_HOME=~/tools/snaphu
export PATH=${PATH}:${SNAPHU_HOME}/bin

```
#### 2. Download

```
cd ~/tools
git clone https://github.com/geodesymiami/MiNoPy.git
git clone https://github.com/insarlab/MintPy.git
git clone https://github.com/yunjunz/PyAPS.git
```

#### 3. install dependencies

Install miniconda if you have not already done so. You may need to close and restart the shell for changes to take effect.
```
# download and install miniconda
# use wget or curl to download in command line or click from the web brower
# MacOS users: curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
# Linux users: curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh

# Mac users:
miniconda_version=Miniconda3-latest-MacOSX-x86_64.sh
# Linux users:
miniconda_version=Miniconda3-latest-Linux-x86_64.sh

wget http://repo.continuum.io/miniconda/$miniconda_version --no-check-certificate -O $miniconda_version
chmod +x $miniconda_version
./$miniconda_version -b -p ~/tools/miniconda3
~/tools/miniconda3/bin/conda init bash
```

Run the following in your terminal to install the dependencies to a new environment minopy (recommended):

```
conda env create -f $MINOPY_HOME/docs/conda_env.yml
conda activate minopy
```
Or run the following in your terminal to install the dependencies to your custom environment, the default is base:

```
conda install --yes -c conda-forge --file ~/tools/MiNoPy/docs/conda.txt
$CONDA_PREFIX/bin/pip install git+https://github.com/insarlab/PySolid.git
$CONDA_PREFIX/bin/pip install git+https://github.com/tylere/pykml.git
```

#### 4. Setup MiNoPy
```
bash $MINOPY_HOME/docs/install
```

#### 5. Install [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) (optional)
```
wget https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.4.tar.gz
tar -xvf snaphu-v2.0.4.tar.gz
mv snaphu-v2.0.4 snaphu
sed -i 's/\/usr\/local/$(SNAPHU_HOME)/g' snaphu/src/Makefile
cd snaphu/src; make

```



### Notes
Please read notes on [PyAPS](https://github.com/yunjunz/PyAPS) and [PySolid](https://github.com/insarlab/PySolid) from [GitHub/MintPy](https://github.com/insarlab/MintPy/blob/main/docs/installation.md) 
