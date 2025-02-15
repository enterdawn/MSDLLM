## MSDLLM
Official PyTorch implementation for the paper - MSDLLM:Multi-modal Image and Text Fusion Sarcasm Detection via Fine-tuning Large Language Model.
### Environment
Python 3.10.8 and `requirement.txt`
### Dataset
Download and unzip the datasets [SarcNet](https://github.com/yuetanbupt/SarcNet) and [MMSD](https://github.com/joeying1019/mmsd2.0).
### Model
Download the model [GLM-4-9B](https://github.com/THUDM/GLM-4) and replace `modeling_chatglm.py` with our's.
### Run
```
python run_sarcnet.py
python run_mmsd.py
```
### Citation
```tex
@article{guan2025msdllm,
  title={MSDLLM:Multi-modal Image and Text Fusion Sarcasm Detection via Fine-tuning Large Language Model},
  author={Zequn Guan and Yongxue Shan and Chengjie Zhou and Yunjiong Liu and Chao Che},
  journal={N/A},
  year={2025}
}
```
### Acknowledgement
Our implementation uses [GLM-4-9B](https://github.com/THUDM/GLM-4) and [TimeLLM](https://github.com/KimMeen/Time-LLM), have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
