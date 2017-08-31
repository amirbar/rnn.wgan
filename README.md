# Language Generation with Recurrent Generative Adversarial Networks without Pre-training

Code for training and evaluation of the model from ["Language Generation with Recurrent Generative Adversarial Networks without Pre-training"](https://arxiv.org/abs/1706.01399).  

Additional Code for using Fisher GAN in Recurrent Generative Adversarial Networks
 
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
GAN_TYPE: String Type of GAN to use. Choose between 'wgan' and 'fgan' for wasserstein and fisher respectively

```

Parameters can be set by either changing their value in the config file or by passing them in the terminal:

```
python curriculum_training.py --START_SEQ=1 --END_SEQ=32
```

## Monitoring Convergence During Training

In the wasserstein GAN, please monitor the disc_cost. It should be a negative number and approach zero. The disc_cost represents the negative wasserstein distance between gen and critic.

## Generating text

The `generate.py` script will generate `BATCH_SIZE` samples using a saved model. It should be run using the parameters used to train the model (if they are different than the default values). For example:

``` 
python generate.py --CKPT_PATH=/path/to/checkpoint --DISC_GRU_LAYERS=2 --GEN_GRU_LAYERS=2
```

## Evaluating text

To evaluate samples using our %-IN-TEST-n metrics, use the following command, linking to a txt file where each row is a sample:

``` 
python evaluate.py --INPUT_SAMPLE=/path/to/samples.txt
```



## Experimental Features (not mentioned in the paper)

To train with fgan with recurrent highway cell:

```
python curriculum_training.py --GAN_TYPE fgan --CRITIC_ITERS 2 --GEN_ITERS 4 \
--PRINT_ITERATION 500 --ITERATIONS_PER_SEQ_LENGTH 60000 --RNN_CELL rhn
```

Please note that for fgan, there may be completely different hyperparameters that are more suitable for better convergence.

### Monitoring Convergence

To measure fgan convergence, gen_cost should start at a positive number and decrease. The lower, the better.

Warning: in the very beginning of training, you may see the gen_cost rise. Please wait at least 5000 iterations and the gen_cost should start to lower. This phenomena is due to the critic finding the appropriate wasserstein distance and then the generator adjusting for it.

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

