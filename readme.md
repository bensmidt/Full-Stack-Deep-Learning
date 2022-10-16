# Full Stack Deep Learning
[Full Stack Deep Learning (FSDL)](https://fullstackdeeplearning.com/course/2022/) is a free public course aimed at teaching people the basics needed to convert their machine learning knowledge to real products. This is not my code but my version of FSDL's repo to play around with and learn on my own. A big thank you to FSDL for making this information freely available!

# About Me
I am a 3rd year at the University of Texas at Austin (hook'em!) studying Electrical and Computer Engineering and Math. I've decided to complete this course (as well as other public courses [EECS498: Deep Learning for Computer Vision](https://github.com/bensmidt/EECS-498-DL-Computer-Vision) and [CS229: Machine Learning](https://github.com/bensmidt/CS229-ML-Autumn-2018)) in an effort to learn about Machine Learning and Deep Learning from a mathematical foundation. 

You can find out more about me at my [LinkedIn](https://www.linkedin.com/in/benjamin-smidt/). If you have any questions about my solutions, learning machine learning, or just want connect, feel free to follow me and reach out on LinkedIn. I can be a little slow to respond depending my current responsiblities but I can assure you I will respond! I hope you enjoy this course as much as I have and that my solutions are helpful :)

Last thing, big thank you to Stanford for making these lectures and course material publically available! They are immensely helpful and provide amazing opportunity to learn from many of the best minds in computer science (and other fields). They've been vital in my machine learning journey. 

# Notes
[Other Resources](https://github.com/bensmidt/Full-Stack-Deep-Learning/blob/master/LecNotes/OtherResources.md)
1. [When to Use ML and Course Vision](https://github.com/bensmidt/Full-Stack-Deep-Learning/blob/master/LecNotes/1-UsingML_CourseVision.md)
2. [Development Infrastructure and Tooling](https://github.com/bensmidt/Full-Stack-Deep-Learning/blob/master/LecNotes/2-DevInfTool.md)

<u>**Everything below is code from FSDL's repository**</u>


# 🥞 Full Stack Deep Learning Fall 2022 Labs

Welcome!

As part of Full Stack Deep Learning 2022, we will incrementally develop a complete deep learning codebase to create and deploy a model that understands the content of hand-written paragraphs.

For an overview of the Text Recognizer application architecture, click the badge below to open an interactive Jupyter notebook on Google Colab:

<div align="center">
  <a href="http://fsdl.me/2022-overview"> <img src=https://colab.research.google.com/assets/colab-badge.svg width=240> </a>
</div> <br>

We will use the modern stack of [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/).

We will use the main workhorses of DL today: CNNs and Transformers.

We will manage our experiments using what we believe to be the best tool for the job: [Weights & Biases](https://docs.wandb.ai/).

We will set up a quality assurance and continuous integration system for our codebase using [pre-commit](https://pre-commit.com/) and [GitHub Actions](https://docs.github.com/en/actions).

We will package up the prediction system and deploy it as a [Docker](https://docs.docker.com/) container on [AWS Lambda](https://aws.amazon.com/lambda/).

We will wrap that prediction system in a frontend written in Python using [Gradio](https://gradio.app/docs).

We will set up monitoring that alerts us to potential issues in our model using [Gantry](https://gantry.io/).

## Click the badges below to access individual lab notebooks on Colab and videos on YouTube

| Lab                                                       | Colab                                            | Video                                                 |
| :--                                                       | :---:                                            | :---:                                                 |
| **Lab Overview**                                          | [![open-in-colab]](https://fsdl.me/lab00-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-overview-video) |
| **Lab 01: Deep Neural Networks in PyTorch**               | [![open-in-colab]](https://fsdl.me/lab01-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-01-video)       |
| **Lab 02a: PyTorch Lightning**                            | [![open-in-colab]](https://fsdl.me/lab02a-colab) | [![yt-logo]](https://fsdl.me/2022-lab-02-video)       |
| **Lab 02b: Training a CNN on Synthetic Handwriting Data** | [![open-in-colab]](https://fsdl.me/lab02b-colab) | [![yt-logo]](https://fsdl.me/2022-lab-02-video)       |
| **Lab 03: Transformers and Paragraphs**                   | [![open-in-colab]](https://fsdl.me/lab03-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-03-video)       |
| **Lab 04: Experiment Tracking**                           | [![open-in-colab]](https://fsdl.me/lab04-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-04-video)       |
| **Lab 05: Troubleshooting & Testing**                     | [![open-in-colab]](https://fsdl.me/lab05-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-05-video)       |
| **Lab 06: Data Annotation**                               | [![open-in-colab]](https://fsdl.me/lab06-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-06-video)       |
| **Lab 07: Deployment**                                    | [![open-in-colab]](https://fsdl.me/lab07-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-07-video)       |
| **Lab 08: Monitoring**                                    | [![open-in-colab]](https://fsdl.me/lab08-colab)  | [![yt-logo]](https://fsdl.me/2022-lab-08-video)       |

[yt-logo]: https://fsdl.me/yt-logo-badge
[open-in-colab]: https://colab.research.google.com/assets/colab-badge.svg
