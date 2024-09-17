# iterative-alignment

* `training` Contains training pipelines for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Note: The Self-Play training pipeline is still under development.
* `inference_and_scoring` contains the modified implementaion from https://github.com/hkust-nlp/deita to score samples on quality and complexity. Added multi-GPU inference support using Hugging Face’s accelerate library.
* To reproduce the environment, use the provided `environment.yml` file
