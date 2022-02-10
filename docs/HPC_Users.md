### This is a manual for HPC users. It is based on job submission in [MinSAR](https://github.com/geodesymiami/rsmas_insar) app

In this workflow, you need to create jobs and then run them in order

To create jobs, run:
```
minopyApp.py $PWD/PichinchaSenDT142.template --dir minopy --jobfiles
```
After the jobs are created, you may run them with one of the appropriate submit commands:
```
submit_jobs.bash $PWD/PichinchaSenDT142.template --minopy
(submit_jobs.bash $PWD/PichinchaSenDT142.template --dostep minopy)
```
`submit_jobs.bash` commmand can be used for running individual jobs in run_files:

```
sbatch_jobs.bash minopy/run_files/run_01_minopy_load_data 
submit_jobs.bash minopy/run_files/run_02_minopy_phase_linking
submit_jobs.bash minopy/run_files/run_03_minopy_concatenate_patch
submit_jobs.bash minopy/run_files/run_04_minopy_generate_ifgram
submit_jobs.bash minopy/run_files/run_05_minopy_unwrap_ifgram
submit_jobs.bash minopy/run_files/run_06_minopy_load_ifgram
submit_jobs.bash minopy/run_files/run_07_mintpy_ifgram_Correction 
submit_jobs.bash minopy/run_files/run_08_minopy_invert_network
submit_jobs.bash minopy/run_files/run_09_mintpy_timeseries_correction
```
