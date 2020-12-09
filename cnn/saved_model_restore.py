import numpy as np
import tensorflow as tf

from tensorflow.python.saved_model import signature_constants

import nvutils
import argparse
import os

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

parser.add_argument('--export_dir', required=True,
                    help="""Directory in which to the saved model.""")

parser.add_argument('--data_dir', default=None,
                    help="""Path to dataset in TFRecord format
                    (aka Example protobufs). Files should be
                    named 'train-*' and 'validation-*'.""")

parser.add_argument('-b', '--batch_size', type=int, 
                    default=128,
                    help="""Size of each minibatch. (Default is 128)""")


parser.add_argument('--display_total', type=int,
                    default=0,
                    help="""How many predictions to print out.
                    (Default is 0 and to print all)""")

FLAGS, unknown_args = parser.parse_known_args()

export_dir = FLAGS.export_dir
data_dir = FLAGS.data_dir
batch_size = FLAGS.batch_size
display_total = FLAGS.display_total

print("Script arguments:")
for flag, val in vars(FLAGS).items():
    if val is not None:
        print("  --{} {}".format(flag, val))

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], export_dir)
    signature = meta_graph_def.signature_def

    graph = tf.get_default_graph()

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    x_tensor_name = signature[signature_key].inputs['input'].name
    c_tensor_name = signature[signature_key].outputs['class_ids'].name
    p_tensor_name = signature[signature_key].outputs['probabilities'].name

    x = graph.get_tensor_by_name(x_tensor_name)
    class_ids = graph.get_tensor_by_name(c_tensor_name)
    probabilities = graph.get_tensor_by_name(p_tensor_name)

    class_ids_, probs_ = None, None
    total = 0
    if data_dir is not None:
        filename_pattern = os.path.join(data_dir, '%s-*')
        eval_filenames  = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

        num_preproc_threads = 10
        dataset = nvutils.image_set(
                        eval_filenames, batch_size, 224, 224,
                        training=False, distort_color=False,
                        deterministic=False,
                        num_threads=num_preproc_threads)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        try:
            while True:
                value_, _ = sess.run(next_element)
                tclass_ids_, tprobs_ = sess.run([class_ids, probabilities], {x: value_})
                total += tclass_ids_.shape[0]
                if class_ids_ is None:
                    class_ids_ = tclass_ids_
                    probs_ = tprobs_
                else:
                    class_ids_ = np.concatenate((class_ids_, tclass_ids_))
                    probs_ = np.concatenate((probs_, tprobs_))
        except tf.errors.OutOfRangeError:
            pass
    else:
        value_ = np.random.random([batch_size, 224, 224, 3]).astype(np.float16)
        class_ids_, probs_ = sess.run([class_ids, probabilities], {x: value_})
        total = batch_size

    for i in range(total if display_total == 0 else display_total):
        class_id = class_ids_[i][0]
        probability = probs_[i][class_id]
        class_name = class_id
        print(i, "Class:", "'" + str(class_name) + "'", "Prob.:", probability)
