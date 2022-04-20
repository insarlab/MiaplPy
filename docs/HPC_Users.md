### This is a manual for HPC users. It is based on job submission in [MinSAR](https://github.com/geodesymiami/rsmas_insar) app

In this workflow, you need to create jobs and then run them in order

To create jobs, run:
```
miaplpyApp.py $PWD/PichinchaSenDT142.template --dir miaplpy --jobfiles
```
After the jobs are created, you may run them with one of the appropriate submit commands:
```
submit_jobs.bash $PWD/PichinchaSenDT142.template --miaplpy
(submit_jobs.bash $PWD/PichinchaSenDT142.template --dostep miaplpy)
```
`submit_jobs.bash` commmand can be used for running individual jobs in run_files:

```
sbatch_jobs.bash miaplpy/{unwrapping_network}/run_files/run_01_miaplpy_load_data 
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_02_miaplpy_phase_linking
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_03_miaplpy_concatenate_patch
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_04_miaplpy_generate_ifgram
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_05_miaplpy_unwrap_ifgram
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_06_miaplpy_load_ifgram
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_07_miaplpy_ifgram_Correction 
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_08_miaplpy_invert_network
submit_jobs.bash miaplpy/{unwrapping_network}/run_files/run_09_miaplpy_timeseries_correction
```
