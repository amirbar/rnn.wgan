# Language Generation with Recurrent Generative Adversarial Networks without Pre-training

## About this work

Generative Adversarial Networks (GANs) have shown great promise recently in image generation. Training GANs for text generation has proven to be more difficult, because of the non-differentiable nature of generating text with recurrent neural networks. Consequently, past work has either resorted to pre-training with maximum-likelihood or used convolutional networks for generation.
In this work, we show that recurrent neural networks can be trained to generate text with GANs from scratch by employing curriculum learning, slowly increasing the length of the generated text, and by training the RNN simultaneously to generate sequences of different lengths. We show that this approach vastly improves the quality of generated sequences compared to the convolutional baseline.
 
### Sample outputs (32 chars)

``` 
" There has been to be a place w
On Friday , the stories in Kapac
From should be taken to make it 
He is conference for the first t
For a lost good talks to ever ti
```

## Training

To start training the network using the CL-VL-TH model, first download the dataset, available at <http://www.statmt.org/lm-benchmark/>, to the `data` directory.

Then use the following command:

```
python curriculum_training.py
```

The following environment is needed:

* Python 2.7
* Tensorflow 1.1
* Scipy
* Matplotlib
* A recent NVIDIA GPU

The following parameters can be configured:

```
LOGS_DIR: Path to save model checkpoints and samples outputs (defaults to './logs/')
DATA_DIR: Path to load the data from (defaults to './data/1-billion-word-language-modeling-benchmark-r13output/')
CKPT_PATH: Path to checkpoint file when restoring a saved model
BATCH_SIZE: Size of batch size (defaults to 64)
CRITIC_ITERS: Number of iterations for the discriminator (defaults to 10)
GEN_ITERS: Number of iterations for the geneartor (defaults to 50)
MAX_N_EXAMPLES: Number of samples to load from dataset (defaults to 10000000)
GENERATOR_MODEL: Name of generator model (currently only 'Generator_GRU_CL_VL_TH' is available)
DISCRIMINATOR_MODEL: Name of discriminator model (currently only 'Discriminator_GRU' is available)
PICKLE_PATH: Path to PKL directory to hold cached pickle files (defaults to './pkl')
ITERATIONS_PER_SEQ_LENGTH: Number of iterations to run per each sequence length in the curriculum training (defaults to 15000)
NOISE_STDEV: Standard deviation for the noise vector (defaults to 10.0)
DISC_STATE_SIZE: Number of neurons for the discriminator GRU (defaults to 512)
GEN_STATE_SIZE: Number of neurons for the genarator GRU (defaults to 512)
TRAIN_FROM_CKPT: Boolean that should hold true when restoring from a saved checkpoint (defaults to false)
GEN_GRU_LAYERS: Number of GRU layers for the genarator (defaults to 1)
DISC_GRU_LAYERS: Number of GRU layers for the discriminator (defaults to 1)
START_SEQ: Sequence length to start the curriculum learning with (defaults to 1)
END_SEQ: Sequence length to end the curriculum learning with (defaults to 32)
SAVE_CHECKPOINTS_EVERY: Save checkpoint every # steps (defaults to 25000)
LIMIT_BATCH: Boolean that indicates whether to randomly pick sequences lengths  (defaults to true)

```

Paramters can be used in the following way:

```
python curriculum_training.py --START_SEQ=1 --END_SEQ=32
```

## Generating text

The `generate.py` script will generate `BATCH_SIZE` samples using a saved model. It should be run in the following way,
using the parameters used to train it:

``` 
python generate.py --CKPT_PATH=/path/to/checkpoint --DISC_GRU_LAYERS=2 --GEN_GRU_LAYERS=2
```

## Evaluating text

To evaluate the output of the generator, use the following command, linking to a txt file where each row is a sample:

``` 
python evaluate.py --INPUT_SAMPLE=/path/to/samples.txt
```

