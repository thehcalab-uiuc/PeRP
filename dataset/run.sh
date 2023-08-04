#!\bin\bash

PCP_PATH="../saved/pcp_models"
offset=2.5
OUTPUT_PATH="./data/offset_$offset"

echo "Starting trajectory collection"
num=0

# simulate and collecting trajectories
for trial in 1 2 3 4 5;
do
    for hc in 1 10 20 50 100;
    do
        folder_name=$PCP_PATH/hc_$hc
        echo "Processing ${folder_name} with process number ${num}"
        python traj_collector.py \
            --run_name run_$num \
            --pcp_path $folder_name \
            --output_path $OUTPUT_PATH/raw \
            --warmup 600 \
            --num_runs 10 \
            --driver_offset $offset \
            --time 4000 
        num=$((num+1))
    done
done

echo "Trajectories collected"

# OUTPUT_PATH/raw now has run_# directories with traj_out directory with the trajectories

echo "Dividing trajectories"
for folder_name in $(ls -A $OUTPUT_PATH/raw);
do
    for trajectory_file in $(ls -A $OUTPUT_PATH/raw/$folder_name/traj_out/);
    do
        echo "processing $OUTPUT_PATH/raw$folder_name/traj_out/$trajectory_file"
        python preprocess.py \
            --input_path $OUTPUT_PATH/raw/$folder_name/traj_out/$trajectory_file \
            --output_path $OUTPUT_PATH/processed/${folder_name} \
            --title_prefix ${folder_name}_${trajectory_file%.*}
    done
done

# OUTPUT_PATH/processed now has all the trajectories divided into the right length passed in, should be 20

echo "Splitting into Train and Test Sets"

DATA_PATH="$OUTPUT_PATH/processed/"
n=$((`find $DATA_PATH -mindepth 1 -maxdepth 2 -type d |wc -l`))
echo "Total: $n"
N_train=$(echo "$n*0.8" |bc -l)
N_train=$(printf "%.*f" 0 $N_train)
echo "Train: $N_train"
let N_val="$n-$N_train"
echo "Val: $N_val"

# ensuring output directory exists
mkdir -p $OUTPUT_PATH/train
mkdir -p $OUTPUT_PATH/val

# combining train
i=0
ls "$DATA_PATH/" |while read folder_name;
do
    if [[ $i < $N_train ]];
    then
        mv "$DATA_PATH/$folder_name/"* $OUTPUT_PATH/train/
    else
        mv "$DATA_PATH/$folder_name/"* $OUTPUT_PATH/val/
    fi
    i=$((i + 1))
done

echo "Train set in $OUTPUT_PATH/train with $(ls $OUTPUT_PATH/train | wc -l) samples"
echo "Val set in $OUTPUT_PATH/val with $(ls $OUTPUT_PATH/val | wc -l) samples"