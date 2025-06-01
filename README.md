## MobiGPT: A Foundation Model for Mobile Wireless Networks

This is the official code of MobiGPT.

### Introduction

MobiGPT is a unified model structure capable of forecasting three types of mobile network data: base station traffic, user app usage, and channel quality. It employs a foundation model architecture that integrates transformer and diffusion mechanisms with domain-aware prompting and task-specific temporal masking. This design enables the model to capture spatiotemporal correlations across heterogeneous data sources and adapt to diverse forecasting tasks—including short-term, long-term, and data generation scenarios. Through evaluations on over 100,000 real-world samples, MobiGPT demonstrates superior performance and generalization across all data types, offering a scalable and transferable solution for future mobile network optimization.

![image-20250601132217696](.\pics\framework.png)

### Requirements

Use python 3.11 from Anaconda with:

```powershell
torch==2.6.0
numpy==1.26.4
timm==1.0.15
diffusers==0.32.2
```

### How to use MobiGPT

##### Train 

```powershell
python main.py --task long_prediction --prompt_state train 
```

You can choose tasks in ['mix', short_prediction', 'long_prediction', 'generation']

##### Test 

```powershell
python main.py --task long_prediction --prompt_state test --save_folder ./yoursavefolder
```

##### zero-shot

```powershell
python main.py --task long_prediction --prompt_state zero-shot --save_folder ./yoursavefolder
```

##### few-shot 

```powershell
python main.py --task long_prediction --prompt_state few-shot --save_folder ./yoursavefolder --fewshot_rate 0.1
```

You can set your few-shot rate.

