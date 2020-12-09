#!/bin/bash

echo -n "Please specify root of dataset directory: "
read DATA_ROOT

mkdir -p ${DATA_ROOT}
if [ $? -ne 0 ]; then
    echo -e "\nFatal: dataset root dir did not exist and couldn't create it\n"
    exit 1
else
    echo -e "\nSuccess: dataset root dir validated\n"
fi

if ! type "wget" > /dev/null; then
    echo -e "\nFatal: wget not available. Install wget and re-run script.\n"
    exit 1
fi

pushd . > /dev/null

cd ${DATA_ROOT}

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz

tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

rm 1-billion-word-language-modeling-benchmark-r13output.tar.gz

popd > /dev/null

FULL_DIR=${DATA_ROOT}/1-billion-word-language-modeling-benchmark-r13output/
cp 1b_word_vocab.txt ${FULL_DIR}

echo -e "\nSuccess! One billion words dataset ready at:"
echo ${FULL_DIR}
echo -e "Please pass this dir to single_lm_train.py via the --datadir option.\n" 
