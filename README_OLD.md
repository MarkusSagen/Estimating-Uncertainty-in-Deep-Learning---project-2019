# Estimating Certainty of Deep Learning

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

## Links

- [Report](https://www.overleaf.com/5521335765qmyyqkdwrjxd)
- [Presentation Slides](https://uppsalauniversitet-my.sharepoint.com/:p:/g/personal/jiahao_lu_2199_student_uu_se1/Eef8bFx6UjxNk-xOMjMLKcABeaJo5fmtrI2GNzQmNgbvvA?e=xXlNhr)
- [OneDrive folder for shared files](https://uppsalauniversitet-my.sharepoint.com/:f:/g/personal/jiahao_lu_2199_student_uu_se1/Eno_J1-CIMpEh-4Yo8J42TwB1tsyL3WjI4xyrvbAC_Rsxg?e=q2ihZJ)
  - [Inter-group OneDrive folder for shared files](https://uppsalauniversitet-my.sharepoint.com/:f:/g/personal/jiahao_lu_2199_student_uu_se1/EqH_UyCfrM9Gn4TRTjtcPzkBa6T9ZzGppPv1CcU9vftLMg?e=F1inP0)
  - [Inter-group repository for shared code snippets](https://github.com/Noodles-321/CSProjectShare)
  - ~~[GoogleDrive](https://drive.google.com/drive/folders/1pkkgM5rkurwRUs3NojLtYgeAv1PlbF0m)~~ (deprecated)
- [Dataset](http://www.cb.uu.se/~joakim/OralData3/OralCancer_DataSet3.zip)
  - Usename: project
  - Password: CancerCells
- [Progress board](https://github.com/Noodles-321/Certainty/projects/1)

## Todo List

- Pre-processing
  
  - [x] Label smoothing
  
- Models & main methods

  |          | Deterministic | Dropout | Concrete Dropout |  VI  | LLVI |
  | :------: | :-----------: | :-----: | :--------------: | :--: | :--: |
  |  LeNet   |       1       |    1    |        1         |  1   |  1   |
  | ResNet50 |       1       |    1    |        1         |  1   |  1   |

- Post-processing (calibration)
  
  - [x] Temperature scaling
  
- Datasets
  - [x] MNIST
  - [x] OralCancer

## Table of Contents

<!-- TOC -->

- [Estimating Certainty of Deep Learning](#estimating-certainty-of-deep-learning)
  - [Links](#links)
  - [Todo List](#todo-list)
  - [Table of Contents](#table-of-contents)
  - [Dependencies](#dependencies)
    - [For running on GPU](#for-running-on-gpu)
    - [For running on CPU](#for-running-on-cpu)
  - [Usage](#usage)
- [Selected Methods](#selected-methods)
  - [Non-Bayesian](#non-bayesian)
    - [MC Dropout based methods](#mc-dropout-based-methods)
  - [Bayesian](#bayesian)
    - [Basic Bayesian](#basic-bayesian)
    - [Sampling based methods](#sampling-based-methods)
    - [Variational Inference based methods](#variational-inference-based-methods)
    - [Optimisers](#optimisers)
  - [Calibration methods](#calibration-methods)
  - [Other tricks](#other-tricks)
  - [Evaluation Metrics](#evaluation-metrics)
    - [General metrics](#general-metrics)
    - [Calibration metrics](#calibration-metrics)
    - [Aleatoric and Epistemic uncertainties](#aleatoric-and-epistemic-uncertainties)
  - [Other things if time permitting](#other-things-if-time-permitting)

<!-- /TOC -->

## Dependencies

- TensorFlow==1.14

### For running on GPU

``` bash
conda create --name tftorch --file requirements.txt
```

or

``` bash
conda install --yes --file requirements.txt
```

or  (for local win-64)

```bash
conda env create -f tftorch.yml
```

### For running on CPU

``` bash
conda create --name certainty_venv python=3.6 --file requirements_CPU.txt -y && conda activate certainty_venv
```

## Usage

To then run the code:

```bash
python framework.py
```

------

# Selected Methods

## Non-Bayesian
### MC Dropout based methods

- [x] Monte Carlo Dropout -- Markus
  - [Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. International Conference on Machine Learning, 1050–1059.](https://arxiv.org/abs/1506.02142)
  - [Code reference](https://github.com/OATML/bdl-benchmarks)
    - [Simplistic Code model implemented](https://github.com/janisgp/Uncertainty_DeepLearning/blob/master/Uncertainty%20on%20MNIST%20using%20Monte%20Carlo%20Dropout.ipynb)
    - [Paper with Code](https://paperswithcode.com/paper/dropout-as-a-bayesian-approximation#code)
    - Ways to extend the simple model: [Way 1 - in pytorch](https://github.com/andyhahaha/Uncertainty-Mnist-with-Pytorch), [Way 2 - in pytorch](https://github.com/xuwd11/Dropout_Tutorial_in_PyTorch)
- [x] Concrete Dropout -- Markus
  - [Gal, Y., Hron, J., Kendall, A. (2017). Concrete Dropout](https://arxiv.org/abs/1705.07832)
  - [Code reference](https://github.com/yaringal/ConcreteDropout/blob/master/spatial-concrete-dropout-keras.ipynb)
- Ensemble Dropout
  - [Smith, L., & Gal, Y. (2018). *Understanding Measures of Uncertainty for Adversarial Example Detection*.](https://arxiv.org/abs/1803.08533v1)
  - [Code reference](https://github.com/OATML/bdl-benchmarks/blob/alpha/baselines/diabetic_retinopathy_diagnosis/deep_ensembles)

## Bayesian

### Basic Bayesian

 - Bayes by Backprop -- Markus
    - [Blundell, C., Cornebise, J., Kavukcuoglu, K., Wierstra, D. : Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf)
    - [Article 1](https://medium.com/neuralspace/bayesian-convolutional-neural-networks-with-bayes-by-backprop-c84dcaaf086e)
    - [Article 2](https://medium.com/neuralspace/probabilistic-deep-learning-bayes-by-backprop-c4a3de0d9743)
    - [Code Availability](http://krasserm.github.io/2019/03/14/bayesian-neural-networks/)
    - [PyTorch code](https://www.nitarshan.com/bayes-by-backprop/)
    - [Bayesian Layer in Keras](https://github.com/bstriner/bayesian_dense)
    - [Code Reference](https://github.com/RobRomijnders/weight_uncertainty/tree/master/weight_uncertainty)
    - [Tensorflow Example](https://github.com/TENGBINN/Bayes-by-Backprop)
    - [Tensorflow Article](https://medium.com/python-experiments/bayesian-cnn-model-on-mnist-data-using-tensorflow-probability-compared-to-cnn-82d56a298f45)
- Bayesian Active Learning 
  - [Code Available](https://github.com/wenqiwooo/Active-Learning-on-Bayesian-Neural-Networks)
  - [Keras Code](https://github.com/wenqiwooo/Active-Learning-on-Bayesian-Neural-Networks/blob/master/experiments/mnist_active_learn.py)
- Bayes by Hypernet
  - [Pawlowski, N., Brock, A., Lee, M., Rajchl, M., Glocker, B. : Implicit Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1711.01297.pdf)
  - [Code Availability](https://github.com/pawni/BayesByHypernet/)
- Reparametrization trick
  - [Kingma, D., Salimans, T., Welling, M. : Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557)

### Sampling based methods
 - Stochastic Hamiltonian Monte Carlo (SHMC)
    - [TODO](todo)
 - Stochastic Gradient Langevin Dynamics (SGLD)
    - [Wang, K., Vicol, P., Lucas, J., Gu, L., Grosse, R., Zemel, R. : Adversarial Distillation of Bayesian Neural Network Posteriors](http://proceedings.mlr.press/v80/wang18i/wang18i.pdf)

### Variational Inference based methods

- [Overview](https://arxiv.org/abs/1901.02731)

- [x] Mean-Field Variational Inference – Lu
  - [Wen, Y., Vicol, P., Ba, J., Tran, D., & Grosse, R. (2018). Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. *ArXiv:1803.04386 [Cs, Stat]*.](http://arxiv.org/abs/1803.04386)
  - [Code reference](https://github.com/OATML/bdl-benchmarks/tree/alpha/baselines/diabetic_retinopathy_diagnosis/mfvi), [blog](https://krasserm.github.io/2019/03/14/bayesian-neural-networks/), [Google's code](https://github.com/google-research/google-research/tree/cbc6b862a9199340d7156d136796c031b7c22c72/uq_benchmark_2019/imagenet)
- Functional
  - [Sun, S., Zhang, G., Shi, J., & Grosse, R. (2019). Functional Variational Bayesian Neural Networks. ArXiv:1903.05779 [Cs, Stat].](http://arxiv.org/abs/1903.05779)
  - [Code reference](https://github.com/ssydasheng/FBNN)
- :star: Adversarial α-divergence Minimization -- Lu
  - [Santana, S. R., & Hernández-Lobato, D. (2019). Adversarial $\alpha$-divergence Minimization for Bayesian Approximate Inference. ArXiv:1909.06945 [Cs, Stat].](http://arxiv.org/abs/1909.06945)
  - [Code reference](https://github.com/simonrsantana/AADM)
- Bayesian CNN with Variational Inference and it's application
  - [Code reference](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)

### Optimisers

- :star: VAdam: Weight-Perturbation -- Markus & Lu
  - [Khan, M. E., Nielsen, D., Tangkaratt, V., Lin, W., Gal, Y., & Srivastava, A. (2018). Fast and scalable Bayesian deep learning by weight-perturbation in Adam. ArXiv Preprint ArXiv:1806.04854.](https://arxiv.org/abs/1806.04854)
  - [Code reference](https://github.com/emtiyaz/vadam)
  - [Code for VOGN](https://github.com/team-approx-bayes/dl-with-bayes)

## Calibration methods

- [x] Temperature Scaling - Lu
  - [Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of the 34th International Conference on Machine Learning-Volume 70, 1321–1330. JMLR. org.](https://arxiv.org/abs/1706.04599)
  - Code reference: [main reference](https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py#L112-L166), [in TensorFlow](https://github.com/google-research/google-research/blob/cbc6b862a9199340d7156d136796c031b7c22c72/uq_benchmark_2019/calibration_lib.py#L31),  [explination](https://geoffpleiss.com/nn_calibration), [simplified code (in PyTorch)](https://github.com/cpark321/uncertainty-deep-learning)
  - **Extensions**:
    - Relaxed SoftMax
      - [Ugo Tanielian, Flavian Vasile. (2019). Relaxed Softmax for learning from Positive and Unlabeled data.](https://arxiv.org/abs/1909.08079)
      - [Code reference](https://github.com/duvenaud/relax/blob/master/relax-autograd/relax.py)
    - :star: Bayesian Temperature Scaling
      - [Max-Heinrich Laves, Sontje Ihler, Karl-Philipp Kortmann, Tobias Ortmaier. (2019). Well-calibrated Model Uncertainty with Temperature Scaling for Dropout Variational Inference.](https://arxiv.org/abs/1909.13550)
      - [Code reference](https://github.com/mlaves/bayesian-temperature-scaling/blob/master/ConfidencePenalty.ipynb)
- [Code reference](https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/Calibration%20-%20(Temp%2C%20Iso%2C%20Beta%2C%20Hist).ipynb) including other methods
  - Beta Calibration
    - [Kull, M., Filho, T. S., & Flach, P. (2017). Beta calibration: A well-founded and easily implemented improvement on logistic calibration for binary classifiers. *Artificial Intelligence and Statistics*, 623–631.](http://proceedings.mlr.press/v54/kull17a.html)
  - Histogram Binning
    - [Zadrozny, B., & Elkan, C. (2001, June 28). *Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers*. 609–616.](http://dl.acm.org/citation.cfm?id=645530.655658)
  - Isotonic Regression
    - [Zadrozny, B., & Elkan, C. (2002, July 23). *Transforming classifier scores into accurate multiclass probability estimates*. 694–699.](https://doi.org/10.1145/775047.775151)
  - Logistic Regression

## Other tricks

- Bootstrap Uncertainty
  - [Barzan Mozafari, Purna Sarkar, Michael Franklin, Michael Jordan, Samuel Madden. (2015). *Scaling Up Crowd-Sourcing to Very Large Datasets:A Case for Active Learning*](https://web.eecs.umich.edu/~mozafari/papers/vldb_2015_crowd.pdf)
  - [Code reference](https://github.com/jmcjacob/Bootstrap_Uncertainty)
  
- [x] Label Smoothing – Lu

  - [Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *ArXiv:1906.02629 [Cs, Stat].](http://arxiv.org/abs/1906.02629)
  
## Evaluation Metrics

### General metrics

- [x] Accuracy
- [x] Average STD of classification
- [x] Average STD of misclassified samples
- [ ] Computation time (training and testing)

### Calibration metrics

- [x] ECE, MCE, Brier
  - [Code reference](https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py)

- [ ] Brier decomposition
  - [Code reference](https://github.com/google-research/google-research/blob/50cb7df528dbed5e7251498df5750ced30bae8e4/uq_benchmark_2019/metrics_lib.py#L265)

- [x] Negative log-likelihood
- [x] AECE, AMCE
  - [Ding, Y., Liu, J., Xiong, J., & Shi, Y. (2019). Evaluation of Neural Network Uncertainty Estimation with Application to Resource-Constrained Platforms.](https://arxiv.org/abs/1903.02050v1)
- Field-ECE -- Markus
  - [Feiyang Pan, Xiang Ao, Pingzhong Tang, Min Lu, Dapeng Liu, Qing He. (2019). Towards reliable and fair probabilistic predictions: field-aware calibration with neural networks.](https://arxiv.org/abs/1905.10713)
  - Code reference: [binning](https://github.com/ondrejba/tf_calibrate), [metrics](https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py)
- [x] Calibration estimating metrics
  - [Vaicenavicius, J., Widmann, D., Andersson, C., Lindsten, F., Roll, J., & Schön, T. B. (2019). Evaluating model calibration in classification. *ArXiv Preprint ArXiv:1902.06977*.](https://arxiv.org/abs/1902.06977)
  - [Code reference](https://github.com/uu-sml/calibration)

### Aleatoric and Epistemic uncertainties

- Definition

  - [Explanation](https://towardsdatascience.com/what-uncertainties-tell-you-in-bayesian-neural-networks-6fbd5f85648e)

  - [Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 5574–5584.](https://arxiv.org/abs/1703.04977)

  - [Kwon, Y., Won, J.-H., Kim, B. J., & Paik, M. C. (2018). *Uncertainty quantification using Bayesian neural networks in classification: Application to ischemic stroke lesion segmentation*.](https://openreview.net/forum?id=Sk_P2Q9sG)

  - ``` python
    prediction = np.mean(p_hat, axis=0)
    aleatoric  = np.mean(p_hat*(1-p_hat), axis=0)
    epistemic  = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2
    ```

  - Code reference for both above: [official](https://github.com/ykwon0407/UQ_BNN), [GitLab in Keras](https://gitlab.com/wdeback/dl-keras-tutorial/blob/master/notebooks/3-cnn-segment-retina-uncertainty.ipynb)

- Softplus Normalisation
  - [Shridhar, K., Laumann, F., & Liwicki, M. (2019). Uncertainty Estimations by Softplus normalization in Bayesian Convolutional Neural Networks with Variational Inference. *ArXiv:1806.05978 [Cs, Stat]*.](http://arxiv.org/abs/1806.05978)
  - [Code reference](https://github.com/kumar-shridhar/PyTorch-Softplus-Normalization-Uncertainty-Estimation-Bayesian-CNN/blob/master/main.ipynb) ([Keras built-in funcation](https://keras.io/activations/) available)
- A comprehensive guide
  - [Shridhar, K., Laumann, F., & Liwicki, M. (2019). A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference. *ArXiv:1901.02731 [Cs, Stat]*.](http://arxiv.org/abs/1901.02731)
  - [Code reference](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)

------

## Other things if time permitting

- [Adversarial Neural network for creating MNIST Adversarial images](https://github.com/hutec/UncertaintyNN/blob/master/notebooks/MNIST%20Adversarial.ipynb)
- Evaluate on notMNIST, expecting a very low confidence.
- [x] RAdam -- Markus
  - [Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J. Han, L. (2019) On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
  - [Code reference](https://github.com/bojone/keras_radam/blob/master/radam.py)
