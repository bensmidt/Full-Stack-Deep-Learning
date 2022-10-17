# [Development Infrastructure and Tooling](https://fullstackdeeplearning.com/course/2022/lecture-2-development-infrastructure-and-tooling/)

## Contents
1. [Reality of ML Systems](#reality-of-ml-systems)
1. [Software Engineering](#software-engineering)
1. [Deep Learning Frameworks](#deep-learning-frameworks)
1. [Meta Frameworks and Model Zoos](#meta-frameworks-and-model-zoos)
1. [Distributed Training](#distributed-training)
1. [Compute](#compute)
1. [Resource Management](#resource-management)
1. [Experiment Management](#experiment-management)
1. [Hyperparameter Optimization](#hyperparameter-optimization)
1. [All-in-one Solutions](#all-in-one-solutions)

&nbsp;

# Reality of ML Systems
1. Collect, aggregate, process, clean, label, and version the data.

1. Find the model architecture and their pre-trained weights and then write and debug the model code.

1. Run training experiments and review the results --> feeds back into trying out new architectures and debugging more code.

1. Deploy the model.

1. Monitor model predictions and close the data flywheel loop. Users generate fresh data for you, which needs to be added to the training set

&nbsp;

# Software Engineering
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

&nbsp;

# Deep Learning Frameworks
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

&nbsp;

# Meta Frameworks and Model Zoos
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

&nbsp;

# Distributed Training
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

- There are more tricks
    - NLP: Alibi, sequence length warmup

- FFCV: research library trying to optimize the data flow

&nbsp;

# Compute
- Compute power needed is unbelievably large 

## Basics
- NVIDIA has been the only choice for awhile
- Google TPUs also nice (GCP cloud only)

## Three Main Factors
1. Amount of data that fits on a GPU
2. Speed of crunching through data on GPU (different for 32bit and 16bit)
3. Speed of communicating between CPU and GPU, and between GPUs

## GPU Comparisons
- New architectures appear almost every year
- Some cards = server use, others = consumer use
- [Lambda Labs GPU Benchmarks](https://lambdalabs.com/gpu-benchmarks)
- [AIME GPU Benchmarks](https://www.aime.info/en/blog/deep-learning-gpu-benchmarks-2021/)
- [FSDL GPU Cloud Comparison](https://fullstackdeeplearning.com/cloud-gpus/)

## Cloud Platforms
- Heavyweights: GCP, AWS, Azure
- Startups: Lambda Labs, Paperspace, Coreweave, etc.

### TPUs
- TPU v4 are fastest possible for deep learning (not GA yet)
- TPU v3 are quite fast and *excel at scaling*
- [FSDL GPU Cloud Comparison](https://fullstackdeeplearning.com/cloud-gpus/)

## Cost and Benchmark Data
- Expensive per-hour DOES NOT EQUAL Expensive per-experiment
- You need to look at (what I'm calling) "compute value"
    - how much compute do you get for $1000 (let's say)
- Heuristic: Use most expensive per-hour GPUs (4x or 8x A100's) 
- Heuristic: Use the least expensive cloud (if possible)
- Startups are much cheaper than the big boys 
    - Price difference = factor of 3

## Your Own GPU
- Building your own 128GP RAM and 2x RTX 3090's - ~$7000
    - One day to build and setup
- Going beyong 4 2000-series or 2 3000-series is painful

- Prebuilt comes at varying costs
- [Lambda Labs GPU Advice](https://lambdalabs.com/blog/best-gpu-2022-sofar/#:~:text=Best%20GPU%20for%20Deep%20Learning%20in%202022%20(so%20far)&text=Lambda%20Scalar%20PCIe%20GPU%20server,NVLink%2C%20NVSwitch%2C%20and%20InfiniBand.)
- [Tim Dettmers on GPUs (2020)](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)

## Recommendations
- Useful to get your own GPU machine 
    - worth it for mindset shift of maximizing utility vs minimizing cost
- Scaling out experiments = cloud
    - TPUs worth experimenting with for large-scale training

&nbsp;

# Resource Management
- Experiment Needs: 
    - Machine(s) + GPU(s)
    - Setup (Python+CUDA version, Python requirements)
    - Data
1. Manual
    - Follow best practices for specifying dependencies (conda and pip tools)
2. SLURM: Cluster of machines
    - Old-school solution to workload management that is still widely use
3. Docker 
    - A way to package up an entire dependency stack in a lighter-than-a-VM package
    - NVIDIA Docker required for GPUs
4. Kubernetes + Kubeflow
    - Kubernetes = way to run many Docker containers on top of a cluster
    - Kubeflow is open source project from Google (no longer controlled by them)
        - Spawn and manage Jupyter notebooks and manage multi-step ML workflows
        - can be useful but it's a lot

- Cloud? 
    - AWS SageMaker
        - Labels data, builds, trains, tunes, and deploys models
        - Notebooks are a key feature
        - Could make sense if you already use AWS for everything
        - increasing support for PyTorch
        - 15-20% markup compared to normal AWS instances
        - Support for spot instances
    - Anyscale
        - Ray Train is new project
        - Claimed to be faster than AWS Sagemaker
        - Intelligent sport instance support
        - Significant markup to AWS
    - Grid.ai (makers of PyTorch Lightning)
        - Not totally sure about the longterm viability of this 
    - Non-Ml specific solutions
        - You can write your own scripts or use some libraries
        - Nothing really recommended 
    - Determined.ai
        - Open-source solution that lets you manage a cluster either on-prem or in the cloud
        - Cluster management + ditributed training
    - A truly simple solution (they think) does not exist

&nbsp;

# Experiment Management
- Tensorboard (not exclusive to TensorFlow)
    - Great solution for single experiment
    - Becomes unwieldly for lots of experiments
- MLFlow 
    - Open source soultion for experiment andmodel management
    - Need to host yourself
- Weights and Biases
    - Hosted free for public projects (paid for private)
    - I think someone who works at FSDL works there
- Determined.ai
- Neptune
- Comet

&nbsp;

# Hyperparameter Optimization
- This software can be very useful 
- Would be great to provide different combinations and stop if underfitting
- Weights and Biases
    - W&B Sweeps
- Other solutions
    - Sagemaker
    - Determined.ai
    - Ray
    - Usually not worth using anything specialized

&nbsp;

# All-in-one Solutions
- Single system for: 
    - Development (hosted notebook)
    - Scaling experiments to many machines 
    - Tracking experiments and versioning models
    - Deploying models
    - Monitoring performance

- Amazon SageMaker
- Gradient from PaperSpace
- Dominoes from Data Labs (meant more for non deep learning machine learning)





    