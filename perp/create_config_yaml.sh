#!/bin/bash

# DATA_PATH="./pcp_models"
DATA_PATH="./results"

# ensuring directory exists
if [ ! -d $DATA_PATH ] 
then
    echo "$DATA_PATH does not exist"
    exit
fi

# If you train a custom PC policy, uncomment and run the below lines:
################### FOR PCP MODELS ###################

# # combining
# for folder_name in $(ls -A $DATA_PATH);
# do
#     log_file=$(find $DATA_PATH/$folder_name -name \*.log -type f | tail -1)
#     echo "Writing from $log_file"
#     if [ ! -z "$log_file" ]
#     then
#         output_file=$DATA_PATH/$folder_name/config.yaml
#         sed -n '1,5p;5q' $log_file > $output_file
#         echo "alg: 'TRPO'" >> $output_file
#         sed -n '7,55p;55q' $log_file >> $output_file
#     fi
# done

######################################################

################### FOR PeRP MODELS ##################

# combining
for folder_name in $(ls -A $DATA_PATH);
do
    output_file=$DATA_PATH/$folder_name/config.yaml
    echo $output_file
    sed -i '' -e "1s/.*/alg: 'TRPO'/" $output_file
done
