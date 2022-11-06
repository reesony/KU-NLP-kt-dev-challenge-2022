# kt-dev-challenge-2022
This is a project that won the first prize at kt-dev-challenge held in September 2022.\
Overall, we used the T5 model provided by KT, but we would like to inform you that we cannot disclose some models and datasets due to confidentiality.

# T5-model architecture
![image](https://user-images.githubusercontent.com/77087144/200172759-0e1d285e-96db-4618-92d6-bbdce03a2a8b.png)
![image](https://user-images.githubusercontent.com/77087144/200172925-2aed22fb-8879-4a96-9a9f-1e590be01c1f.png)

# Configuration of Implementation
* [How to preprocess](#How-to-preprocess)
* [How to train](#How-to-train)
* [How to find best model](#How-to-find-best-model)
* [How to infer](#How-to-infer)
* [How to start](#How-to-start)

---

## How to preprocess
preprocessor.py 
* To use T5 Encoder only model, the sentence must be changed to a tensor.
* This is used to transform from sentence to tensor.

---

## How to train
train.py
* This is used to train T5 encoder-decoder model.
* Cannot disclose train code for encoder only models due to confidentiality. 

---

## How to find best model
search_hyperparams.py
* This is used to search hyperparameters to optimize model.
* Hyperparameter search borrowed the idea of the grid search method.
search_ensemble_f1.py
* This is used to search optimal combination.
* The method searching best combination used the idea of the hard-voting.

---

## How to infer
infer.py and infer_encoder.py
* This is used to infer results using dataset of test.
* infer.py is used to infer results using T5 encoder-decoder model
* infer_encoder.py is used to infer results using T5 encoder only model. But due to confidentiality, this code is not perfect code.

---

## How to start

### Clone project and install modules


