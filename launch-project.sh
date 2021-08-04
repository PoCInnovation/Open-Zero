#!/bin/bash

save="save.pt"

usage ()
{
    echo -e "usage:"
    echo -e "\t${0} <-m [mode]> [option]\n"
    echo -e "mode:\n"
    echo -e "\ttrain\t\tTrain mode, will launch the training process of the AI."
    echo -e "\ttest\t\tTest mode, will launch the testing process of the AI.\n"
    echo -e "option:\n"
    echo -e "\t -h\t\tHelp, print this message."
    echo -e "\t -s <file>\tSave, specify a save file (default: 'save.pt')."
    exit
}

train_mode ()
{
    python3 src/chess_a3c.py ${save}
    exit ${?}
}

test_mode ()
{
    python3 src/chess_ai_validation.py &
    python3 src/solo_play.py ${save}
}

while getopts "hs:m:" option; do
    case ${option} in
        h) usage ;;
        m) mode=${OPTARG} ;;
        s) save=${OPTARG} ;;
        :) usage ;;
        *) usage ;;
esac done

[ -z "${mode}" ] && usage ;
case ${mode} in
    "train") train_mode ;;
    "test") test_mode ;;
    *) usage ;;
esac

