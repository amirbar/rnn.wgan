import numpy as np
import model_and_data_serialization
from model import *

output_path = './output/sample.txt'

'''
example usage:
python generate.py --GENERATOR_MODEL=Generator_GRU_CL_VL_TH --DISCRIMINATOR_MODEL=Discriminator_GRU --CKPT_PATH=path/to/checkpoint
'''

SEQ_LEN = FLAGS.END_SEQ

_, charmap, inv_charmap = model_and_data_serialization.load_dataset()
charmap_len = len(charmap)

Generator = get_generator(GENERATOR_MODEL)
Discriminator = get_discriminator(DISCRIMINATOR_MODEL)

_, inference_op = Generator(BATCH_SIZE, charmap_len, seq_len=SEQ_LEN)
disc_fake = Discriminator(inference_op, charmap_len, SEQ_LEN, reuse=False)

saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, CKPT_PATH)
    sequential_output, scores = session.run([inference_op, disc_fake])

samples = []

for i in range(BATCH_SIZE):
    chars = []
    for seq_len in range(sequential_output.shape[1]):
        char_index = np.argmax(sequential_output[i,seq_len])
        chars.append(inv_charmap[char_index])
    sample = "".join(chars)
    samples.append(sample)

if not(os.path.isdir('./output')):
    os.mkdir("./output")

with open(output_path, 'w') as f:
    for k in samples:
        f.write("%s\n"%k)
f.close()
print "Prediction saved to: %s"%output_path