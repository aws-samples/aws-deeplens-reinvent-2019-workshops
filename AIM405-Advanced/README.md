AIM405 - Optimize deep learning models for edge deployments with AWS DeepLens
======================================================================================

Abstract
--------

In this workshop, learn how to train and optimize your computer vision pipelines for edge deployments with Amazon SageMaker Ground Truth and AWS DeepLens. Also learn how to build a sample image classification model with Amazon SageMaker with GluonCV and deploy it to AWS DeepLens. Finally, learn how to optimize your deep learning models and code to achieve faster performance for use cases where speed matters.

Presenter: Nathaniel Slater - Senior Manager, Phu Nguyen - Product Manager

<a href="https://aws.amazon.com/"><img src="_static/aws_logo.png" alt="AWS Icon" height="45"></a> &nbsp; <a href="https://ml.aws/"><img src="_static/amazon_ai.png" alt="AmazonAI Icon" height="58"></a> &nbsp; <a href="https://mxnet.incubator.apache.org/"><img src="_static/apache_incubator_logo.png" alt="Apache Incubator Icon" height="39"></a> &nbsp; <a href="https://mxnet.incubator.apache.org/"><img src="_static/mxnet_logo_2.png" alt="MXNet Icon" height="39"></a> &nbsp; <a href="https://gluon-cv.mxnet.io/"><img src="_static/gluon_logo_horizontal_small.png" alt="Gluon Icon" height="42"></a> 

Agenda
------

| Time        | Title                                                        | Notebooks |
| ----------- | ------------------------------------------------------------ | --------- |
| 10:45-00:00 | ..                                                           |           |
| 10:45-00:00 | Lab 0 - Lab environment, SageMaker Notebook instance, set up | [link][0] |
| 10:45-00:00 | Lab 1 - Labeling dataset using Amazon SageMaker Ground Truth | [link][1] |
| 10:45-00:00 | Lab 2 - Train an image classification model using GluonCV    | [link][2] |
| 10:45-00:00 | Lab 3 - Deploy custom model to AWS DeepLens                  | [link][3] |
<!---| 10:45-00:00 | Lab 4 - Optimizing models with Amazon SageMaker Neo          | [link][4] |--->
| 12:00-13:00 | Q&A and Closing                                              |           |

Q&A
---
Q1: How do I setup the environment for this workshop?

A1: We recommend to use Amazon SageMaker notebook instance. Or you can clone this GIT repository into anywhere as long as all the required libraries such as Apache MXNet, GluonCV, and AWS SDK(Boto3) can be installed. 

**Please follow [the lab setup guide](./Lab0/setup.md) to launch your Amazon SageMaker notebook instance for this workshop.**

Q2: How can I train a model using Amazon SageMaker built-in image classification algorithm and deploy it to AWS DeepLens?

A2: Refer to [AIM229 - Start using computer vision with AWS DeepLens][5] workshop material for the detail.


Authors
---

<span style="color:grey">Jiyang Kang, Muhyun Kim, Nathaniel Slater, Phu Nguyen, Tatsuya Arai</span>

[0]: ./Lab0/setup.md
[1]: ./Lab1/deeplens-l400-lab1-gt.ipynb
[2]: ./Lab2/lab2-image-classification.ipynb
[3]: ./Lab3/README.md
[4]: ./Lab4/lab4-neo.ipynb
[5]: ../AIM229-Beginner
