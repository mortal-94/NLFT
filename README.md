# NLFT: Natural Language Fine-Tuning

This repo contains minimal source code and data to reproduce the results in the research paper NLFT: Natural Language Fine-Tuning

## Minimal Instruction

### Clone Our Repo

```bash
git clone https://github.com/Julia-LiuJ/NLFT
cd NLFT
```

### Environment Setup (pip or conda)

```bash
pip install -r requirements.txt # python 3.9+
conda create -f nlft.yaml
```

### Download LLaMA3 8B (from e.g. modelscope)

```bash
python utils/llama_download.py
wandb login
```

### Run scipts

#### NLFT

```bash
# pip install transformers==4.41.1
sh finetune-correct.sh
```

#### SFT

```bash
# pip install transformers==4.37.0
sh finetune-sft.sh
```

#### Evaluation

```bash
# pip install transformers==4.41.1
sh submit.sh
```

## License

应该就是 Apache License 2.0


## Citation

If you find this code useful in your research, please consider citing our paper:

```

```