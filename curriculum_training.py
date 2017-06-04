import model_and_data_serialization
from config import *
from single_length_train import run
from summaries import log_run_settings

create_logs_dir()
log_run_settings()

_, charmap, inv_charmap = model_and_data_serialization.load_dataset(seq_length=32, b_lines=False)

REAL_BATCH_SIZE = FLAGS.BATCH_SIZE

if FLAGS.SCHEDULE_SPEC == 'all' :
    stages = range(FLAGS.START_SEQ, FLAGS.END_SEQ)
else:
    split = FLAGS.SCHEDULE_SPEC.split(',')
    stages = map(int, split)

print('@@@@@@@@@@@ Stages : ' + ' '.join(map(str, stages)))

for i in range(len(stages)):
    prev_seq_length = stages[i-1] if i>0 else 0
    seq_length = stages[i]
    print(
    "**********************************Training on Seq Len = %d, BATCH SIZE: %d**********************************" % (
    seq_length, BATCH_SIZE))
    tf.reset_default_graph()
    if FLAGS.SCHEDULE_ITERATIONS:
        iterations = min((seq_length + 1) * FLAGS.SCHEDULE_MULT, FLAGS.ITERATIONS_PER_SEQ_LENGTH)
    else:
        iterations = FLAGS.ITERATIONS_PER_SEQ_LENGTH
    run(iterations, seq_length, seq_length == stages[0] and not (FLAGS.TRAIN_FROM_CKPT),
        charmap,
        inv_charmap,
        prev_seq_length)

    if FLAGS.DYNAMIC_BATCH:
        BATCH_SIZE = REAL_BATCH_SIZE / seq_length
