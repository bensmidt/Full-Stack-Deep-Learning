# [Development Infrastructure and Tooling](https://fullstackdeeplearning.com/course/2022/lecture-2-development-infrastructure-and-tooling/)

## Contents
1. [Reality of ML Systems](#reality-of-ml-systems)
1. [Software Engineering](#software-engineering)
1. [Deep Learning Frameworks](#deep-learning-frameworks)
1. [Meta Frameworks and Model Zoos](#meta-frameworks-and-model-zoos)
1. [Distributed Training](#distributed-training)

## Reality of ML Systems
1. Collect, aggregate, process, clean, label, and version the data.

1. Find the model architecture and their pre-trained weights and then write and debug the model code.

1. Run training experiments and review the results --> feeds back into trying out new architectures and debugging more code.

1. Deploy the model.

1. Monitor model predictions and close the data flywheel loop. Users generate fresh data for you, which needs to be added to the training set

## Software Engineering
- Programming Language: Python is the clear winner
- Code Editor: VSCode is highly recommended
- Notebooks: 
    - Jupyter Notebooks are very popular but have problems (although there are people and companies who work around this)
        - primitive code editor
        - our of order execution artifacts
        - hard to version
        - hard to test
    - Write code in a script --> reload into notebook
- Streamlit: very convenient for working with python
- Environment: 
    - Specify Python and CUDA versions in `environment.yml`
    - Use the `conda` package manger to create environment from this file
    - Specify requirements in `requirement/prod.in` and `requirements/dev.in`
    - Use `pip-tools` to lock in mutually compatible versions of all requirements
    - Add `Makefile` to simply run `make` to update everything

## Deep Learning Frameworks
Why DL frameworks? Deep Learning is not much code (provided you have a matrix math library). Auto-differentiation and CUDA are a lot of work and layer types, optimizers, data interfaces, etc. complicate things.

1. PyTorch (Recommended):
    - PyTorch is absolutely dominant
        - by # models, # papers, # competition
    - Make faster with: TorchScript
    - Great distributed training ecosystem
    - Abundant libraries
    - Mobile deployment
    - PyTorch Lightning is great
1. TensorFlow: 
    - Tensorflow.js --> run ML models in browser
    - Keras is "unmatched in easily creating models"
1. Jax: 
    - Not specific to deep learning
    - For general vectorization, auto-differentiation, and compilation to GPU/TPU code
    - Requires separate framework for deep learning (Flax or Haiku)
1. FastAI
    - Provides many advanced tricks
        - data augmentation, better initialization, learning rate scheduling
    - Code style is very different from mainstream Python

## Meta Frameworks and Model Zoos
Many times you can use a pretrained model instead of from scratch
1. ONNX
    - Open standard for saving deep learning models
    - Can convert PyTorch -> TensorFlow, etc.
    - Can work well, but you may run into edge cases
1. Hugging Face
    - Most populat Model Zoo: 60K pre-trained models
    - Transformers library that works with PyTorch, TF, Jax
    - 7.5 Datasets
1. TIMM (CV)
    - Collection of state of the art CV models

## Distributed Training
Multiple machines with multiple GPUs. There are two major storage components: data batch and model parameters
1. Trivial Parallelism: 
    - Data batch fits on single GPU
    - Model parameters fit on single GPU
2. [Data Parallelism](https://openai.com/blog/techniques-for-training-large-neural-networks/) 
    - Data batch doesn't fit on single GPU
    - Model parameters fit on single GPU
    - Distribute a single batch of data across GPUs --> average gradients across them
    - GPUs need to have fast interconnect (NVLink, etc)
    - Speedup
        - Linear for server cards
        - Sublinear for consumer cards
    - PyTorch has robust DistibutedDataParallel library
        - Can also use Horovod (third-party library)
        - Easily use either with PyTorch Lightning 
3. [Sharded Data-Parallelism](https://arxiv.org/abs/1910.02054)
    
    What exactly is in GPU memory? 
    - Model params, gradients, optimizer states (Adam), batch of data

    Solution: 
    - 01: Shard optimizer states
    - 02: Shard optimizer states, gradients
    - 03: Shard optimizer states, gradients, model parameters

    Model parameters are literally passed between GPUs as computation proceeds!

    [Helpful Article/Video](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

    Implementations
    - Deepspeed: Microsoft
    - Fairscale: Facebook
    - Pytorch --> Fully-sharded DataParallel

    You can apply ZeRO-Offload to single GPU if you wish
    - Fairscale calls is CPU-offloading

4. Pipelined Model-Parallelism
    - Essentially you give a single layer to a single GPU
    - Problem: only one GPU runs at one time
        - DeepSpeed and FairScale implement this in a manner that reduces this issue
    - This method requires tuning the amount of pipelining
    - NOT a fire and forget solution (like Sharded-Data Parallelism

5. Tensor Parallelism
    - Distribute computation of tensors across GPUs

6. All of them!
    You can use all of these 
    ZeRO (Fully Sharded) data paralelism works much better than it used to though so using 






## GPUs


    