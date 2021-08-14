#! /bin/bash
#set -x

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
helptext="                                                                       \n\
  Examples:                                                                      \n\
      minopyApp.bash  $TE/GalapagosSenDT128.template                             \n\
      minopyApp.bash  $TE/GalapagosSenDT128.template --dostep crop               \n\
      minopyApp.bash  $TE/GalapagosSenDT128.template --start  invesion           \n\
      minopyApp.bash  $TE/GalapagosSenDT128.template --stop   ifgrams            \n\
      minopyApp.bash  $TE/GalapagosSenDT128.template                             \n\
                                                                                 \n\
  Processing steps (start/end/dostep): \n\
   Command line options for steps processing with names are chosen from the following list: \n\
                                                                                 \n\
   ['crop', 'inversion', 'ifgrams', 'unwrap', 'mintpy_corrections']              \n\
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
    PROJECT_NAME=$(basename "$1" | cut -d. -f1)
    exit_status="$?"
    if [[ $PROJECT_NAME == "" ]]; then
       echo "Could not compute basename for that file. Exiting. Make sure you have specified an input file as the first argument."
       exit 1;
    fi
fi
template_file=$1
if [[ $PROJECT_NAME == "minopy_template.cfg" ]]; then
  template_real_path = $(realpath "$1")
  WORK_DIR=$( dirname "$template_real_path")
else
  WORK_DIR=$SCRATCHDIR/$PROJECT_NAME/minopy
fi

mkdir -p $WORK_DIR
cd $WORK_DIR

echo "$(date +"%Y%m%d:%H-%m") * `basename "$0"` $@ " >> "${WORK_DIR}"/log

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

if [[ ${#POSITIONAL[@]} -gt 1 ]]; then
    echo "Unknown parameters provided."
    exit 1;
fi

crop_flag=1
inversion_flag=1
ifgrams_flag=1
unwrap_flag=1
mintpy_corrections_flag=1

if [[ $startstep == "inversion" ]]; then
    crop_flag=0
elif [[ $startstep == "ifgrams" ]]; then
    crop_flag=0
    inversion_flag=0
elif [[ $startstep == "unwrap" ]]; then
    crop_flag=0
    inversion_flag=0
    ifgrams_flag=0
elif [[ $startstep == "mintpy_corrections" ]]; then
    crop_flag=0
    inversion_flag=0
    ifgrams_flag=0
    unwrap_flag=0
elif [[ $startstep != "" ]]; then
    echo "startstep received value of "${startstep}". Exiting."
    exit 1
fi

if [[ $stopstep == "crop" ]]; then
    inversion_flag=0
    ifgrams_flag=0
    unwrap_flag=0
    mintpy_corrections_flag=0
elif [[ $stopstep == "inversion" ]]; then
    ifgrams_flag=0
    unwrap_flag=0
    mintpy_corrections_flag=0
elif [[ $stopstep == "ifgrams" ]]; then
    unwrap_flag=0
    mintpy_corrections_flag=0
elif [[ $stopstep == "unwrap" ]]; then
    mintpy_corrections_flag=0
elif [[ $stopstep != "" ]]; then
    echo "stopstep received value of "${stopstep}". Exiting."
    exit 1
fi

####################################
minopyApp_cmd=`which minopyApp.py`

if [[ $crop_flag == "1" ]]; then
    cmd="$minopyApp_cmd $template_file --dostep crop --job"
    echo " Running.... python $cmd >& out_minopy_crop.e &"
    python $cmd >& out_minopy_crop.e &
    echo "$(date +"%Y%m%d:%H-%m") * $cmd" >> "${WORKDIR}"/log

    cmd="submit_minopy_jobs.bash $template_file --dostep crop"
    echo "Running.... $cmd"
    $cmd
    exit_status="$?"
    if [[ $exit_status -ne 0 ]]; then
       echo "submit_jobs.bash --dostep crop  exited with a non-zero exit code ($exit_status). Exiting."
       exit 1;
    fi
fi


if [[ $inversion_flag == "1" ]]; then
    cmd="$minopyApp_cmd $template_file --dostep inversion --job"
    echo " Running.... python $cmd >& out_minopy_inversion.e &"
    python $cmd >& out_minopy_inversion.e &
    echo "$(date +"%Y%m%d:%H-%m") * $cmd" >> "${WORKDIR}"/log

    cmd="submit_minopy_jobs.bash $template_file --dostep inversion"
    echo "Running.... $cmd"
    $cmd
    exit_status="$?"
    if [[ $exit_status -ne 0 ]]; then
       echo "submit_jobs.bash --dostep inversion  exited with a non-zero exit code ($exit_status). Exiting."
       exit 1;
    fi
fi

if [[ $ifgrams_flag == "1" ]]; then
    cmd="$minopyApp_cmd $template_file --dostep ifgrams --job"
    echo " Running.... python $cmd >& out_minopy_ifgrams.e &"
    python $cmd >& out_minopy_ifgrams.e &
    echo "$(date +"%Y%m%d:%H-%m") * $cmd" >> "${WORKDIR}"/log

    cmd="submit_minopy_jobs.bash $template_file --dostep ifgrams"
    echo "Running.... $cmd"
    $cmd
    exit_status="$?"
    if [[ $exit_status -ne 0 ]]; then
       echo "submit_jobs.bash --dostep ifgrams  exited with a non-zero exit code ($exit_status). Exiting."
       exit 1;
    fi
fi

if [[ $unwrap_flag == "1" ]]; then
    cmd="$minopyApp_cmd $template_file --dostep unwrap --job"
    echo " Running.... python $cmd >& out_minopy_unwrap.e &"
    python $cmd >& out_minopy_unwrap.e &
    echo "$(date +"%Y%m%d:%H-%m") * $cmd" >> "${WORKDIR}"/log

    cmd="submit_minopy_jobs.bash $template_file --dostep unwrap"
    echo "Running.... $cmd"
    $cmd
    exit_status="$?"
    if [[ $exit_status -ne 0 ]]; then
       echo "submit_jobs.bash --dostep unwrap  exited with a non-zero exit code ($exit_status). Exiting."
       exit 1;
    fi
fi


if [[ mintpy_corrections_flag == "1" ]]; then
    cmd="$minopyApp_cmd $template_file --dostep write_correction_job --job"
    echo " Running.... python $cmd >& out_mintpy_corrections.e &"
    python $cmd >& out_mintpy_corrections.e &
    echo "$(date +"%Y%m%d:%H-%m") * $cmd" >> "${WORKDIR}"/log

    cmd="submit_minopy_jobs.bash $template_file --dostep mintpy_corrections"
    echo "Running.... $cmd"
    $cmd
    exit_status="$?"
    if [[ $exit_status -ne 0 ]]; then
       echo "submit_jobs.bash --dostep mintpy_corrections  exited with a non-zero exit code ($exit_status). Exiting."
       exit 1;
    fi
fi
