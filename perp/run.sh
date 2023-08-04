#!/bin/bash

export FLOW_RES_DIR="./FLOW_RES_DIR"

RESULTS_DIR="./results"

mkdir -p $RESULTS_DIR

PCP_PATH="../saved/pcp_models/*/"
TRAIT_INFERENCE_PATH="../saved/trait_2.5"
learning_rate=0.0001

for folder_name in $(ls -d $PCP_PATH);
do
	for i in 0 1 2 3 4
	do
		trial_num=$(find ./$RESULTS_DIR/trial_* -maxdepth 0 -type d | wc -l | sed 's/ //g') 
		mkdir -p "./$RESULTS_DIR/trial_$trial_num/"
		
		echo "Processing for $folder_name save in $RESULTS_DIR/trial_$trial_num w lr=$learning_rate"
		
		python ./pexps/ring.py "./$RESULTS_DIR/trial_$trial_num" \
			e=False \
			warmup_steps=600 \
			horizon=4000 \
			n_steps=100 \
			result_save=results.csv \
			gamma=0.99 \
			alg='TRPO' \
			use_critic=False \
			demand_schedule=cosine \
			demand_amplitude=5 \
			demand_frequency=2 \
			render=False \
			carla=False \
			use_ray=False \
			vae_trait_inference=True \
			trait_inference_path=$TRAIT_INFERENCE_PATH \
			simple_trait_inference=False \
			pcp_path=$folder_name \
			lr=$learning_rate \
			map_path="../saved/map_files/ring.net.xml" \
        	route_path="../saved/map_files/ring.route.xml" \
			beta_speed=1 \
			beta_error=1
	done
done
