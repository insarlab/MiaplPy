### This is a manual for HPC users. It is based on job submission in [MinSAR](https://github.com/geodesymiami/rsmas_insar) app

In this workflow, you need to create jobs and then run them in order

To create jobs, run:
```
minopyApp.py $PWD/PichinchaSenAT18.template --dir minopy --job
```
After the jobs are created, you may run them with appropriate submit commands. MinSAR provides `sbatch_conditional.bash` to further 
limit number of submitting jobs:

```
sbatch_conditional.bash minopy/run_files/run_01_minopy_crop 
sbatch_conditional.bash minopy/run_files/run_02_minopy_inversion
sbatch_conditional.bash minopy/run_files/run_03_minopy_ifgrams
sbatch_conditional.bash minopy/run_files/run_04_minopy_un-wrap
sbatch_conditional.bash minopy/run_files/run_05_mintpy_corrections
```
