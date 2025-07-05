#export CUDA_VISIBLE_DEVICES="0"


# ====== Settings for LlaMa-2 model of three scales =====
model="/models/Llama-2-7b-chat-hf"
early_exit_layers="16,18,20,22,24,26,28,30,32"

# model="/models/Llama-2-13b-chat-hf"
# early_exit_layers="20,22,24,26,28,30,32,34,36,38,40"

# model="/models/llama-2-70b-chat-hf"
# early_exit_layers="60,62,64,66,68,70,72,74,76,78,80"


# set your dataset path here
data_path="../data/trqa"
relative_top="0.01"
max_token='50'
alpha="1"
model_name=$(basename $model)


# END
output_path="../Result/OG/${model_name}_${alpha}_${relative_top}.json"
 python ../tfqa_eval.py --model-name $model --early-exit-layers $early_exit_layers --data-path $data_path --output-path $output_path --num-gpus 1  --relative_top $relative_top --alpha $alpha --max-new-tokens ${max_token}


# baseline
#output_path="../Result/OG/${model_name}_baseline.json"
#python ../tfqa_eval.py --model-name $model  --data-path $data_path --output-path $output_path --num-gpus 1

