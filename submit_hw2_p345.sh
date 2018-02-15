#!/bin/bash

read -p "Please enter your SUNetID: " sunetid
read -p "Enter the location of your catkin_workspace (to default to $HOME/catkin_ws/, press Enter): " catkin_ws
catkin_ws=${catkin_ws:-$HOME/catkin_ws/}

rm -f "${sunetid}_hw2_p345.zip"
echo "Creating ${sunetid}_hw2_p345.zip"
zip -q "${sunetid}_hw2_p345.zip" "submit_hw2_p345.sh"
zip -qd "${sunetid}_hw2_p345.zip" "submit_hw2_p345.sh" # making an empty zip file

for fname in "Problem_3/P3_SVM.ipynb" \
             "Problem_4/detect.py" \
             "${catkin_ws}/src/asl_turtlebot/scripts/detector.py" \
             "${catkin_ws}/src/asl_turtlebot/scripts/pose_controller.py" \
             "${catkin_ws}/src/asl_turtlebot/scripts/supervisor.py"
do
    if [ -f $fname ]; then
        zip "${sunetid}_hw2_p345.zip" $fname
    else
        read -p "$fname not found. Skip it? [yn]: " yn
        case $yn in
            [Yy]* ) ;;
            * ) exit;;
        esac
    fi
done

if [ -f "$sunetid.pdf" ]; then
    zip "${sunetid}_hw2_p345.zip" "$sunetid.pdf"
else
    echo "Cannot find ./$sunetid.pdf; you must submit your HW2 writeup as $sunetid.pdf in this directory."
    exit
fi

echo ""
echo "Querying Stanford server (AFS) to determine submission number; enter SUNetID password if prompted."
ssh_result=$(ssh -o NumberOfPasswordPrompts=1 -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/%r@%h:%p $sunetid@cardinal.stanford.edu ls -t /afs/ir/class/aa274/HW2 2>/dev/null)
ssh_success=$?
lastsub=$(echo ${ssh_result} | tr ' ' '\n' | grep -m 1 ${sunetid}_hw2_p345_submission_[0-9]*.zip | grep -Eo 'submission_[0-9]{1,4}' | grep -Eo '[0-9]{1,4}') # very unorthodox
if [ $ssh_success -eq "255" ]; then
    tput setaf 1
    tput bold
    echo "
             / ██████╗ ███████╗████████╗██████╗ ██╗   ██╗██╗
  \    /\   |  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝██║
   )  ( ') <   ██████╔╝█████╗     ██║   ██████╔╝ ╚████╔╝ ██║
  (  /  )   |  ██╔══██╗██╔══╝     ██║   ██╔══██╗  ╚██╔╝  ╚═╝
   \(__)|   |  ██║  ██║███████╗   ██║   ██║  ██║   ██║   ██╗
             \ ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝
An error occurred when connecting to the submission server. Please retry.
(One cause of this error message is entering your password incorrectly.)"
    exit
fi

subnum=$((lastsub + 1))
echo "Copying to AFS (running command below); enter SUNetID password if prompted."
set -x
scp -o NumberOfPasswordPrompts=1 -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/%r@%h:%p "${sunetid}_hw2_p345.zip" "$sunetid@cardinal.stanford.edu:/afs/ir/class/aa274/HW2/${sunetid}_hw2_p345_submission_$subnum.zip" 2>/dev/null
ssh_success=$?
{ set +x; } 2>/dev/null

if [ $ssh_success -eq "255" ]; then
    tput setaf 1
    tput bold
    echo "
             / ██████╗ ███████╗████████╗██████╗ ██╗   ██╗██╗
  \    /\   |  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝██║
   )  ( ') <   ██████╔╝█████╗     ██║   ██████╔╝ ╚████╔╝ ██║
  (  /  )   |  ██╔══██╗██╔══╝     ██║   ██╔══██╗  ╚██╔╝  ╚═╝
   \(__)|   |  ██║  ██║███████╗   ██║   ██║  ██║   ██║   ██╗
             \ ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝
An error occurred when connecting to the submission server. Please retry.
(One cause of this error message is entering your password incorrectly.)"
    exit
else
    tput setaf 2
    tput bold
    echo "
             / ███████╗██╗   ██╗ ██████╗ ██████╗███████╗███████╗███████╗██╗
  \    /\   |  ██╔════╝██║   ██║██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝██║
   )  ( ') <   ███████╗██║   ██║██║     ██║     █████╗  ███████╗███████╗██║
  (  /  )   |  ╚════██║██║   ██║██║     ██║     ██╔══╝  ╚════██║╚════██║╚═╝
   \(__)|   |  ███████║╚██████╔╝╚██████╗╚██████╗███████╗███████║███████║██╗
             \ ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝"
fi
