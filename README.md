
# Multi-Teacher Knowledge Distillation

This repository contains a part of the code developed as part of my [PhD dissertation](https://arxiv.org/abs/2302.07215): "Multi-teacher knowledge distillation as an effective method for compressing ensembles of neural networks".

## Abstract

Deep learning has contributed greatly to many successes in artificial intelligence in recent years.
Breakthroughs have been made in many applications such as natural language processing, speech processing, or computer vision. Recently, many techniques have been implemented to enable the training of increasingly larger and deeper neural networks.
Today, it is possible to train models that have thousands of layers and hundreds of billions of parameters on powerful GPU or TPU clusters. 
Large-scale deep models have achieved great success, but the enormous computational complexity and gigantic storage requirements make it extremely difficult to implement them in real-time applications, especially on devices with limited resources, which is very common in the case of offline inference on edge devices.

On the other hand, the size of the dataset is still a real problem in many domains. Data are often missing, too expensive, or impossible to obtain for other reasons. Ensemble learning is partially a solution to the problem of small datasets and overfitting. By training many different models on subsets of the training set, we are able to obtain more accurate and better-generalized predictions. However, ensemble learning in its basic version is associated with a linear increase in computational complexity. 
We check if there are methods based on Ensemble learning which, while maintaining the generalization increase characteristic of ensemble learning, will be immediately more acceptable in terms of computational complexity. As part of this work, we analyze the various aspects that influence ensemble learning: We investigate methods of quick generation of submodels for ensemble learning, where multiple checkpoints are obtained while training the model only once.  We analyzed the impact of the ensemble decision-fusion mechanism and checked various methods of sharing the decisions including voting algorithms. Finally, we used the modified knowledge distillation framework as a decision-fusion mechanism which allows the addition compressing of the entire ensemble model into a  weight space of a single machine learning model.

We showed that knowledge distillation can aggregate knowledge from multiple teachers in only one student model and, with the same computational complexity, obtain a better-performing model compared to a model trained in the standard manner. We have developed our own method for mimicking the responses of the teacher's models using a student model, which consists of imitating several teachers at the same time, simultaneously. 

We tested these solutions on several benchmark datasets. In the end, we presented a wide application use of the efficient multi-teacher knowledge distillation framework, with a description of dedicated software developed by us and examples of two implementations.
In the first example, we used knowledge distillation to develop models that could automate corrosion detection on aircraft fuselage. The second example describes the detailed implementation process of the technology in a solution enabling the detection of smoke on observation cameras in order to counteract wildfires in forests.

## Files

[config/const.py](./config/const.py) - Constant value used in the project, folder locations.
<br />
[config/datasets.py](./config/datasets.py) - Parameters defining the datasets used in the research, such as image size or number of classes, I also defined the image augmentation policy here.
<br />
[config/models.py](./config/models.py) - Parameters defining the base models of convolutional networks used in the research, I also defined here the parameters defining the training process, such as the number of epochs or the learning rate.

[scripts/generate_subset.py](./scripts/generate_subset.py) - Helper script generating a CSV file containing paths to a selected files in the dataset. In my research, I checked how the size of training subset of data affects teacher performance, diversity between teachers and the performance of the entire ensemble model and the final student model.
<br />
[scripts/generate_subset.py](./scripts/train_val_test_split.py) - Helper script that divides the data into training, validation and test parts.

[teachers/train.py](./teachers/train.py) - Wrapper class that automates the training of teacher models.

[distillation/train.py](./distillation/train.py) - It contains classes that allow to create a student model and train it by knowledge distillation from many teachers.

[notebooks/results.ipynb](./notebooks/results.ipynb) - Jupyter notebook in which the results and figures of the obtained test results are generated.
<br />
[notebooks/plots.ipynb](./notebooks/plots.ipynb) - Jupyter notebook in which some figures included in the methodology chapter are generated.

[Files](https://drive.google.com/drive/folders/1QnY-zEwSccwxK84wuWNDJPF-7r4gjtM1?usp=sharing) that describe how data samples were split down into subsets.
