## Install MiaplPy

#### 1. Download source code
```
cd ~/tools
git clone https://github.com/insarlab/MiaplPy.git
cd ~/tools/MiaplPy
```

#### 2. Install dependencies
```
mamba env create --file conda-env.yml
```
or if you have an existing environment:

```
mamba env update --name my-existing-env --file conda-env.yml
```

#### 3. Install MiaplPy via pip
```
conda activate miaplpy-env
python -m pip install .
```

#### 4. Install [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/)
```
export TOOLS_DIR=~/tools
cd ~/tools;
wget --no-check-certificate  https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.5.tar.gz
tar -xvf snaphu-v2.0.5.tar.gz
mv snaphu-v2.0.5 snaphu;
rm snaphu-v2.0.5.tar.gz;
sed -i 's/\/usr\/local/$(TOOLS_DIR)\/snaphu/g' snaphu/src/Makefile
cd snaphu/src; make
export PATH=${TOOLS_DIR}/snaphu/bin:${PATH}
```

#### Notes

Please read notes on the account setup for [PyAPS](https://github.com/insarlab/pyaps#2-account-setup-for-era5).
