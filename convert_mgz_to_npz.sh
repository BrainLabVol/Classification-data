#!/bin/bash

length=$#
anat=""

counter=0
str1=${1}
export SUBJECTS_DIR=$2
export FREESURFER_HOME=$1
source $FREESURFER_HOME/SetUpFreeSurfer.sh



#transform mgz and convert mgz to nii
f="${3}.nii.gz"
echo ""
mri_convert $4 $2/$3"/mri/"$f --apply_transform $2/$3/mri/transforms/talairach.xfm --devolvexfm $3 -ic 0 0 0


#mv "${2}/${3}/Convert_msg_report.txt" "${6}/"

python3 $5 "${2}/${3}/mri" $3 $6
str2="${2}/${3}/mri/${f}"
rm "${str2}" 
