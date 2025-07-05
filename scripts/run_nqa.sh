# export CUDA_VISIBLE_DEVICES="0"

model="/data/models/Llama-2-7b-chat-hf"
data_path="../data/"
dataset="triviaqa"
# "natural_questions" or "triviaqa"

early_exit_layers="16,18,20,22,24,26,28,30,32"
relative_top="0.01"
max_token='30'
alpha="3"
model_name=$(basename $model)


# END
output_path="../Result/QA/${model_name}_${alpha}_${relative_top}.json"
python ../qa_eval.py --model-name $model --early-exit-layers $early_exit_layers --dataset_name $dataset --data-path $data_path --output-path $output_path --num-gpus 1 --do-rating --relative_top $relative_top --alpha $alpha --max-new-tokens $max_token 


# baseline
#output_path="../Result/QA/${model_name}_baseline.json"
# python qa_eval.py --model-name $model --dataset_name $dataset --data-path $data_path --output-path $output_path --num-gpus 1 --do-rating 

