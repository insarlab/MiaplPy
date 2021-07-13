### This is a manual for HPC users. It is based on job submission in [MinSAR](https://github.com/geodesymiami/rsmas_insar) app

In this workflow, you need to create jobs and then run them in order

To create jobs, run:
```
minopyApp.py $PWD/PichinchaSenAT18.template --dir minopy --job
```
After the jobs are created, you may run them with one of the appropriate submit commands:
```
submit_jobs.bash $PWD/PichinchaSenAT18.template --minopy
submit_jobs.bash $PWD/PichinchaSenAT18.template --dostep minopy
```
This uses the `sbatch_conditional.bash` commmand, that can be used for individual run_files:

```
sbatch_conditional.bash minopy/run_files/run_01_minopy_load_slc 
sbatch_conditional.bash minopy/run_files/run_02_minopy_inversion
sbatch_conditional.bash minopy/run_files/run_03_minopy_ifgrams
sbatch_conditional.bash minopy/run_files/run_04_minopy_un-wrap
sbatch_conditional.bash minopy/run_files/run_05_mintpy_corrections
```
