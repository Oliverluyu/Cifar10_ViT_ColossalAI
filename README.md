## Project Objective

This project prompts students to acquire hands-on experience with Colossal-AI, which is a powerful tool for distributed training of large-scale deep learning models. 
Taking advantage of Colossal-AI, users are enabled to greatly speed up training and inferencing their custom deep learning model. Let us have a quick look of how to use a few lines of commands to train the model with distributed training empowered.

## New Dataset

The Colossal-AI demo code for training a Vision Transformer, a dataset including 1300 healthy and ill bean leaves images named 'beans' is used. However, this project would like to explore the performance of Colossal-AI distributed training with a larger dataset with around 3000 images. ['dog_food'](url), which contains three distinct classes: chicken, muffin and dog, are retrieved from Hugging Face dataset.

## ViT Model

The original ViT model was introduced in the [paper](url) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al. The [ViT](url) model used in this project is pre-trained on ImageNet-21k at resolution 224x224, an open-sourced model from Hugging Face library. 

## Colossal-AI Plugin

ViT training module is adapted with ColossalAI Boosting API loaded with a plugin chosen from TorchDDPPlugin, GeminiPlugin, LowLevelZeroPlugin and HybridParallelPlugin, which different parallel training strategies. Fine-tuning ViT model in this project leverages Colab environment with only a single GPU, though it could not fully demonstrate the Colossal-AI's robust multi-GPU distributed training advantage, it still offers a general view of the effectiveness of the plugins.

## ViT Performance on 'dogfood' Dataset 

Pretrained ViT utilizes Gemini plugin for fine-tuning of 10 epochs. Here is the training loss and accuracy chart.


## Different Plugins Benchmark Comparison


