#!/bin/bash

RESULTS_DIR="./results"
TRAIT_INFERENCE_PATH="../saved/trait_2.5"

for folder_name in $(ls -A $RESULTS_DIR);
do  
    pcp_path=$(sed '13!d' $RESULTS_DIR/$folder_name/config.yaml)
    pcp_path=$(echo $pcp_path | cut -d ":" -f 2 | sed 's/^[ \t]*//;s/[ \t]*$//')
    echo $pcp_path

    python pexps/ring.py "$RESULTS_DIR/$folder_name" \
        e=True \
        warmup_steps=600 \
        horizon=4000 \
        n_steps=100 \
        result_save="$RESULTS_DIR/$folder_name/eval.csv" \
        use_critic=False \
        render=False \
        carla=False \
        use_ray=False \
        vae_trait_inference=True \
        trait_inference_path=$TRAIT_INFERENCE_PATH \
        simple_trait_inference=False \
        pcp_path="./$pcp_path"
done