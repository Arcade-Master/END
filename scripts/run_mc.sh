# export CUDA_VISIBLE_DEVICES="2" 

model="/data/models/Llama-2-7b-chat-hf"
early_exit_layers="16,18,20,22,24,26,28,30,32"

data_path="../data/trqa"
max_gpu_memory="65"
relative_top="0"
alpha="0.25"
model_name=$(basename $model)

# END 
output_path="../Result/MC/${model_name}_${alpha}_${relative_top}.json"
python ../tfqa_mc_eval.py --model-name $model --early-exit-layers $early_exit_layers --data-path $data_path --output-path $output_path --num-gpus 1 --relative_top $relative_top --alpha $alpha 

# Baseline
# output_path="../Result/MC/${model_name}_baseline.json"
# python ../tfqa_mc_eval.py --model-name $model  --data-path $data_path --output-path $output_path --num-gpus 1 --relative_top $relative_top --alpha $alpha

