##! /bin/bash

function abbreviate {
    abb=$1
    if [[ "${#abb}" -gt $2 ]]; then
        abb=$(echo "$(echo $(basename $abb) | cut -c -$3)...$(echo $(basename $abb) | rev | cut -c -$4 | rev)")
    fi
    echo $abb
}

function remove_from_list {
    var=$1
    shift
    list=("$@")
    new_list=() # Not strictly necessary, but added for clarity

    #echo "VAR: $var"
    for item in ${list[@]}
    do
        #echo "$item"
        if [ "$item" != "$var" ]
        then
            new_list+=("$item")
        fi
    done
    list=("${new_list[@]}")
    unset new_list
    echo "${list[@]}"
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
helptext="                                                                                       \n\
Job submission script                                                                            \n\
usage: submit_minopy_jobs.bash custom_template_file [--start] [--stop] [--dostep] [--help]       \n\
                                                                                                 \n\
  Examples:                                                                                      \n\
      submit_minopy_jobs.bash \$SAMPLESDIR/unittestGalapagosSenDT128.template                    \n\
      submit_minopy_jobs.bash \$SAMPLESDIR/unittestGalapagosSenDT128.template --start ifgrams    \n\
      submit_minopy_jobs.bash \$SAMPLESDIR/unittestGalapagosSenDT128.template --dostep crop      \n\
      submit_minopy_jobs.bash \$SAMPLESDIR/unittestGalapagosSenDT128.template --stop unwrap      \n\

                                                                                 \n\
 Processing steps (start/end/dostep):                                            \n\
                                                                                 \n\
   ['crop', 'inversion', 'ifgrams', 'unwrap', 'mintpy_corrections' ]             \n\
                                                                                 \n\
   In order to use either --start or --dostep, it is necessary that a            \n\
   previous run was done using one of the steps options to process at least      \n\
   through the step immediately preceding the starting step of the current run.  \n\
                                                                                 \n\
   --start STEP          start processing at the named step [default: load_data].\n\
   --end STEP, --stop STEP                                                       \n\
                         end processing at the named step [default: upload]      \n\
   --dostep STEP         run processing at the named step only                   \n
     "
    printf "$helptext"
    exit 0;
else
    BASE_NAME=$(basename "$1" | cut -d. -f1)
    exit_status="$?"
    if [[ $BASE_NAME == "minopy_template" ]]; then
       echo "Could not find template file. Exiting. Make sure you have specified minopy_template.cfg as the first argument."
       exit 1;
    fi
fi
template_real_path = $(realpath "$1")
WORK_DIR=$( dirname "$template_real_path")
RUNFILES_DIR=$WORKDIR"/run_files"

cd $WORKDIR

startstep=1
stopstep=5

start_datetime=$(date +"%Y%m%d:%H-%m")
echo "${start_datetime} * submit_jobs.bash ${WORKDIR} ${@:2}" >> "${WORKDIR}"/log

while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --start)
            startstep="$2"
            shift # past argument
            shift # past value
            ;;
	--stop)
            stopstep="$2"
            shift
            shift
            ;;
	--dostep)
            startstep="$2"
            stopstep="$2"
            shift
            shift
            ;;
        *)
            POSITIONAL+=("$1") # save it in an array for later
            shift # past argument
            ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

#find the last job (11 for 'geometry' and 16 for 'NESD')
job_file_arr=(run_files/run_*_0.job)
last_job_file=${job_file_arr[-1]}
last_job_file_number=${last_job_file:14:2}

if [[ $startstep == "crop" ]]; then
    startstep=1
elif [[ $startstep == "inversion" ]]; then
    startstep=2
elif [[ $startstep == "ifgrams" ]]; then
    startstep=3
elif [[ $startstep == "unwrap" ]]; then
    startstep=4
elif [[ $startstep == "mintpy_corrections" ]]; then
    startstep=5
fi

if [[ $stopstep == "crop" ]]; then
    stopstep=$last_job_file_number
elif [[ $stopstep == "timeseries" ]]; then
    stopstep=$((last_job_file_number+1))
elif [[ $stopstep == "insarmaps" ]]; then
    stopstep=$((last_job_file_number+2))
fi

for (( i=$startstep; i<=$stopstep; i++ )) do
    stepnum="$(printf "%02d" ${i})"
    if [[ $i -le $last_job_file_number ]]; then
	fname="$RUNFILES_DIR/run_${stepnum}_*.job"
    elif [[ $i -eq $((last_job_file_number+1)) ]]; then
	fname="$WORKDIR/smallbaseline_wrapper*.job"
    else
	fname="$WORKDIR/insarmaps*.job"
    fi
    globlist+=("$fname")
done


for g in "${globlist[@]}"; do
    files=($g)
    echo "Jobfiles to run: ${files[@]}"

    # Submit all of the jobs and record all of their job numbers
    jobnumbers=()
    for (( f=0; f < "${#files[@]}"; f++ )); do
	file=${files[$f]}

        jobnumber=$(submit_job_conditional $file)
        exit_status="$?"
        if [[ $exit_status -eq 0 ]]; then
            jobnumbers+=("$jobnumber")
        else
            echo "Couldnt submit job (${file}), because there are $MAX_JOBS_PER_QUEUE active jobs right now. Waiting 5 minutes to submit next job."
             f=$((f-1))
             sleep 300 # sleep for 5 minutes
        fi

    done

    echo "Jobs submitted: ${jobnumbers[@]}"
    sleep 5
    # Wait for each job to complete
    for (( j=0; j < "${#jobnumbers[@]}"; j++)); do
        jobnumber=${jobnumbers[$j]}

        # Parse out the state of the job from the sacct function.
        # Format state to be all uppercase (PENDING, RUNNING, or COMPLETED)
        # and remove leading whitespace characters.
        state=$(sacct --format="State" -j $jobnumber | grep "\w[[:upper:]]\w")
        state="$(echo -e "${state}" | sed -e 's/^[[:space:]]*//')"

        # Keep checking the state while it is not "COMPLETED"
        secs=0
        while true; do

            # Only print every so often, not every 30 seconds
            if [[ $(( $secs % 30)) -eq 0 ]]; then
                echo "$(basename $WORKDIR) $(basename "$file"), ${jobnumber} is not finished yet. Current state is '${state}'"
            fi

            state=$(sacct --format="State" -j $jobnumber | grep "\w[[:upper:]]\w")
            state="$(echo -e "${state}" | sed -e 's/^[[:space:]]*//')"

                # Check if "COMPLETED" is anywhere in the state string variables.
                # This gets rid of some strange special character issues.
            if [[ $state == *"TIMEOUT"* ]] && [[ $state != "" ]]; then
                jf=${files[$j]}
                init_walltime=$(grep -oP '(?<=#SBATCH -t )[0-9]+:[0-9]+:[0-9]+' $jf)

                echo "${jobnumber} timedout due to too low a walltime (${init_walltime})."

                # Compute a new walltime and update the job file
                update_walltime.py "$jf" &> /dev/null

                updated_walltime=$(grep -oP '(?<=#SBATCH -t )[0-9]+:[0-9]+:[0-9]+' $jf)

                datetime=$(date +"%Y-%m-%d:%H-%M")
                echo "${datetime}: re-running: ${jf}: ${init_walltime} --> ${updated_walltime}" >> "${RUNFILES_DIR}"/rerun.log

                echo "Resubmitting file (${jf}) with new walltime of ${updated_walltime}"

                # Resubmit as a new job number
                jobnumber=$(submit_job_conditional $jf)
                exit_status="$?"
		if [[ $exit_status -eq 0 ]]; then
		    jobnumbers+=("$jobnumber")
		    files+=("$jf")
		    echo "${jf} resubmitted as jobumber: ${jobnumber}"
		else
		    echo "sbatch re-submit error message: $jobnumber"
                    echo "sbatch re-submit error: exit code $exit_status. Exiting."
                    exit 1
		fi

                break;

	    elif [[ $state == *"COMPLETED"* ]] && [[ $state != "" ]]; then
                state="COMPLETED"
                echo "${jobnumber} is complete"
                break;

            elif [[ ( $state == *"FAILED"* ) &&  $state != "" ]]; then
                echo "${jobnumber} FAILED. Exiting with status code 1."
                exit 1;
            elif [[ ( $state ==  *"CANCELLED"* ) &&  $state != "" ]]; then
                echo "${jobnumber} was CANCELLED. Exiting with status code 1."
                exit 1;
            fi

            # Wait for 30 second before chcking again
            sleep 30
            ((secs=secs+30))

            done

        echo Job"${jobnumber} is finished."

    done

    # Run check_job_output.py on all files
    cmd="check_job_outputs.py  ${files[@]}"
    echo "$cmd"
    $cmd
       exit_status="$?"
       if [[ $exit_status -ne 0 ]]; then
            echo "Error in submit_jobs.bash: check_job_outputs.py exited with code ($exit_status)."
            exit 1
       fi
done