# czech_gec

This branch [MasterThesis_PechmanPetr_2024](https://github.com/petrpechman/czech_gec/tree/MasterThesis_PechmanPetr_2024) is an implementation of Master's thesis, **Czech Grammar Error Correction** by Pechman Petr.

All experiments, training, and evaluation of models were run on a Kubernetes cluster. We use Kubernetes deployments to run experiments; each deployment needs a Docker image in which all the necessary tools are already installed.

Here's how to run experiments:

### 1. Docker image

We need to build a Docker image that will be used by our Kubernetes deployment. An already prepared script can be used to create an image:
```bash
IMAGE_NAME="<docker-image-name>"
bash build.sh $IMAGE_NAME
```

### 2. Configuration
To run the experiment, we need to have several configuration files that define the training and evaluation of the models. Such configuration files must be in the folder that is stored in `code/src` (in the Docker image). An example of such a configuration is, for example, `code/src/transformer`. The [`transformer`](./code/src/transformer/) folder contains config.json, where all training and evaluation parameters are described, including paths to data, parallelization size. It also contains `errors_config.json`, which defines the probabilities of typical Czech errors that are artificially introduced into the data, and `f_score_dev.json`, where the best achieved F<sub>0.5</sub>-score is continuously written.

### 3. Kubernetes Deployment
Example deployments for model training and evaluation are stored in the [`kubernetes`](./kubernetes/) folder. Deployments always run a python script [run.py](./code/src/pipeline/run.py) which starts the relevant training or evaluation.

### Others
In the [utils](./code/src/utils/) folder, there are necessary scripts that are used in training or evaluation, but there are also scripts for initializing the model, creating a tokenizer, using the [Aspell](http://aspell.net/) or [MorpohoDiTa](https://ufal.mff.cuni.cz/morphodita) tools.