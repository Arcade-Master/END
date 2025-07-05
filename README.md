# Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models

This repository contains the official implementation of the paper:

**"Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models"**

📄 **Paper**: [NAACL 2025 Findings](https://aclanthology.org/2025.findings-naacl.217.pdf)

## Overview

This work introduces a novel decoding method that improves the factuality of large language models by utilizing token-wise cross-layer entropy. Our approach analyzes the internal representations across different layers to identify and mitigate factual errors during text generation.

## Project Structure

```
END-main/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
├── data/                       # Data directory (empty - see setup instructions)
├── scripts/                    # Evaluation scripts
│   ├── run_tfqa.sh            # TruthfulQA open-ended evaluation
│   ├── run_mc.sh              # TruthfulQA multiple choice evaluation  
│   └── run_nqa.sh             # Natural Questions & TriviaQA evaluation
├── transformers-4.42.3/       # Modified transformers library
├── decoding.py                 # Core decoding implementation
├── factor_eval.py              # Factor evaluation utilities
├── GPT3_rating.py              # GPT-3 based evaluation
├── qa_eval.py                  # Question answering evaluation
├── tfqa_eval.py                # TruthfulQA evaluation
├── tfqa_mc_eval.py             # TruthfulQA multiple choice evaluation
└── trivia_eval_util.py         # TriviaQA evaluation utilities
```

## Setup

### Environment Setup

```bash
# Create conda environment
conda create --name decode python=3.8
conda activate decode

# Install modified transformers library
pip install -e transformers-4.42.3

# Install other dependencies
pip install -r requirements.txt
```

### Data Setup

⚠️ **Important**: You need to download the evaluation datasets before running experiments.

The test data for all benchmarks are publicly available:
- **TruthfulQA**: [https://github.com/sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- **Natural Questions**: [https://ai.google.com/research/NaturalQuestions](https://ai.google.com/research/NaturalQuestions)
- **TriviaQA**: [https://nlp.cs.washington.edu/triviaqa/](https://nlp.cs.washington.edu/triviaqa/)

Please refer to the respective papers and repositories for detailed data preparation instructions.

## Usage

### 🔧 Before Running

**You must configure the data and model paths in the scripts before running any experiments.**

Check the script files in the `scripts/` directory and update the paths according to your setup.

### TruthfulQA Evaluation

#### Multiple Choice

```bash
cd scripts
bash run_mc.sh
```

#### Open-ended Generation

```bash
cd scripts
bash run_tfqa.sh
```

### TruthfulQA GPT-3 Evaluation Setup

For open-ended TruthfulQA evaluation, you need to fine-tune two GPT-3 models using OpenAI API:

```bash
# Fine-tune for truthfulness evaluation
openai api fine_tunes.create -t finetune_truth.jsonl -m davinci-002 -n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1

# Fine-tune for informativeness evaluation  
openai api fine_tunes.create -t finetune_info.jsonl -m davinci-002 --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
```

**Note**: In our work, we used an enhanced version of GPT-3 (Davinci-002) for evaluation. You need to:
1. Set your OpenAI API key in `GPT3_rating.py`
2. Update the fine-tuned model IDs in the evaluation script

### Natural Questions & TriviaQA Evaluation

```bash
cd scripts
bash run_nqa.sh
```

## Key Components

- **`decoding.py`**: Core implementation of our token-wise cross-layer entropy decoding method
- **`factor_eval.py`**: Utilities for evaluating factual accuracy
- **`GPT3_rating.py`**: GPT-3 based automatic evaluation for TruthfulQA
- **`qa_eval.py`**: General question answering evaluation framework
- **`tfqa_eval.py`** & **`tfqa_mc_eval.py`**: TruthfulQA specific evaluation scripts
- **`trivia_eval_util.py`**: TriviaQA evaluation utilities

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{wu-etal-2025-improve-decoding,
    title = "Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models",
    author = "Wu, Jialiang  and  Shen, Yi  and  Liu, Sijia  and  Tang, Yi  and  Song, Sen  and  Wang, Xiaoyi  and  Cai, Longjun",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    year = "2025",
    url = "https://aclanthology.org/2025.findings-naacl.217/"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this work, please open an issue in this repository or contact the authors.

