## Project Objective

This project prompts students to acquire hands-on experience with Colossal-AI, which is a powerful tool for distributed training of large-scale deep learning models. 
Taking advantage of Colossal-AI, users are enabled to greatly speed up training and inferencing their custom deep learning model. Let us have a quick look of how to use a few lines of commands to train the model with distributed training empowered.

## New Dataset

The Colossal-AI demo code for training a Vision Transformer, a dataset including 1300 healthy and ill bean leaves images named 'beans' is used. However, this project would like to explore the performance of Colossal-AI distributed training with a larger dataset with around 3000 images. ['dog_food'](url), which contains three distinct classes: chicken, muffin and dog, are retrieved from Hugging Face dataset.

## ViT Model

The original ViT model was introduced in the [paper](url) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al. The [ViT](url) model used in this project is pre-trained on ImageNet-21k at resolution 224x224, an open-sourced model from Hugging Face library. 

## Colossal-AI Plugin

ViT training module is adapted with ColossalAI Boosting API loaded with a plugin chosen from TorchDDPPlugin, GeminiPlugin, LowLevelZeroPlugin and HybridParallelPlugin, which are different parallel training strategies in Colossal-AI. Fine-tuning ViT model in this project leverages Colab environment with only a single GPU, though it could not fully demonstrate the Colossal-AI's robust multi-GPU distributed training advantage, it still offers a general view of the effectiveness of the plugins.

## ViT Performance on 'dogfood' Dataset 

Pretrained ViT utilizes Gemini plugin for fine-tuning of 10 epochs. Here is the training loss and accuracy chart. You may notice that ViT model quickly converge after only 4 epochs. 

![image](https://github.com/Oliverluyu/Cifar10_ViT_ColossalAI/assets/57708978/dbbab642-05b2-4873-ae6a-9a9f7ba1462e)

## Different Plugins Benchmark Comparison

### Throughput for Plugins

Throughput denotes the number of batches processed per second, indicating four plugins performance across different batch sizes. low_level_zero and torch_ddp_fp16 plugins have high throughput across batch sizes and Gemini also has relatively high throughput as batch size increases. torch_ddp, however, does not have a higher throughput when batch size goes up.

![image](https://github.com/Oliverluyu/Cifar10_ViT_ColossalAI/assets/57708978/5712e2db-d8a2-43f4-a8c5-fb99d34e880e)


### Memory Usage per GPU for Plugins

torch_ddp has the highest memory occupation per GPU compared to other plugins, while Gemini can be considered memory efficient as it only has less than a half memory usage per GPU in comparison with the former. The memory usage for Gemini plugin are less than 1GB per GPU when batch size is smaller than 64 and it merely reaches 1.6 GB with batch size climbing up to 128.

![image](https://github.com/Oliverluyu/Cifar10_ViT_ColossalAI/assets/57708978/e774d838-6c82-4fda-a1e4-dfb412a347b1)
