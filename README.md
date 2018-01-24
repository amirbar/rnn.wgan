# Language Generation with Recurrent Generative Adversarial Networks without Pre-training

Code for training and evaluation of the model from ["Language Generation with Recurrent Generative Adversarial Networks without Pre-training"](https://arxiv.org/abs/1706.01399).  


 
### Sample outputs (32 chars)

``` 
" There has been to be a place w
On Friday , the stories in Kapac
From should be taken to make it 
He is conference for the first t
For a lost good talks to ever ti
```

## Training

To start training the CL+VL+TH model, first download the dataset, available at <http://www.statmt.org/lm-benchmark/>, and extract it into the `./data` directory.

Then use the following command:

```
python curriculum_training.py
```

The following packages are required:

* Python 2.7
* Tensorflow 1.1
* Scipy
* Matplotlib


The following parameters can be configured:

```
LOGS_DIR: Path to save model checkpoints and samples during training (defaults to './logs/')
DATA_DIR: Path to load the data from (defaults to './data/1-billion-word-language-modeling-benchmark-r13output/')
CKPT_PATH: Path to checkpoint file when restoring a saved model
BATCH_SIZE: Size of batch (defaults to 64)
CRITIC_ITERS: Number of iterations for the discriminator (defaults to 10)
GEN_ITERS: Number of iterations for the geneartor (defaults to 50)
MAX_N_EXAMPLES: Number of samples to load from dataset (defaults to 10000000)
GENERATOR_MODEL: Name of generator model (currently only 'Generator_GRU_CL_VL_TH' is available)
DISCRIMINATOR_MODEL: Name of discriminator model (currently only 'Discriminator_GRU' is available)
PICKLE_PATH: Path to PKL directory to hold cached pickle files (defaults to './pkl')
ITERATIONS_PER_SEQ_LENGTH: Number of iterations to run per each sequence length in the curriculum training (defaults to 15000)
NOISE_STDEV: Standard deviation for the noise vector (defaults to 10.0)
DISC_STATE_SIZE: Discriminator GRU state size (defaults to 512)
GEN_STATE_SIZE: Genarator GRU state size (defaults to 512)
TRAIN_FROM_CKPT: Boolean, set to True to restore from checkpoint (defaults to False)
GEN_GRU_LAYERS: Number of GRU layers for the genarator (defaults to 1)
DISC_GRU_LAYERS: Number of GRU layers for the discriminator (defaults to 1)
START_SEQ: Sequence length to start the curriculum learning with (defaults to 1)
END_SEQ: Sequence length to end the curriculum learning with (defaults to 32)
SAVE_CHECKPOINTS_EVERY: Save checkpoint every # steps (defaults to 25000)
LIMIT_BATCH: Boolean that indicates whether to limit the batch size  (defaults to true)

```

Parameters can be set by either changing their value in the config file or by passing them in the terminal:

```
python curriculum_training.py --START_SEQ=1 --END_SEQ=32
```

## Generating text

The `generate.py` script will generate `BATCH_SIZE` samples using a saved model. It should be run using the parameters used to train the model (if they are different than the default values). For example:

``` 
python generate.py --CKPT_PATH=/path/to/checkpoint/seq-32/ckp --DISC_GRU_LAYERS=2 --GEN_GRU_LAYERS=2
```

(If your model has not reached stage 32 in the curriculum, make sure to change the '32' in the path above to the maximal stage in the curriculum that your model trained on.)

## Evaluating text

To evaluate samples using our %-IN-TEST-n metrics, use the following command, linking to a txt file where each row is a sample:

``` 
python evaluate.py --INPUT_SAMPLE=/path/to/samples.txt
```


## Reference
If you found this code useful, please cite the following paper:

```
@article{press2017language,
  title={Language Generation with Recurrent Generative Adversarial Networks without Pre-training},
  author={Press, Ofir and Bar, Amir and Bogin, Ben and Berant, Jonathan and Wolf, Lior},
  journal={arXiv preprint arXiv:1706.01399},
  year={2017}
}
```

## Acknowledgments

This repository is based on the code published in [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training).
