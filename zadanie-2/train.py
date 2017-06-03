# # Typical setup to include TensorFlow.
# import datetime
# import tensorflow as tf
#
# LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y")
#
# # Make a queue of file names including all the JPEG images files in the relative
# # image directory.
# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("/data/spacenet2/images/*.jpg"))
#
# # Read an entire image file which is required since they're JPEGs, if the images
# # are too large they could be split in advance to smaller files or use the Fixed
# # reader to split up the file.
# image_reader = tf.WholeFileReader()
#
# # Read a whole file from the queue, the first returned value in the tuple is the
# # filename which we are ignoring.
# _, image_file = image_reader.read(filename_queue)
#
# # Decode the image as a JPEG file, this will turn it into a Tensor which we can
# # then use in training.
# image = tf.image.decode_jpeg(image_file)
# image_batch = tf.expand_dims(image, 0)
# summary_op = tf.summary.image("plot", image_batch)
# all_summaries = tf.summary.merge_all()
#
# # Start a new session to show example output.
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
#
#     # Required to get the filename matching to run.
#     tf.initialize_all_variables().run()
#
#     # Coordinate the loading of image files.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     # Get an image tensor and print its value.
#     # image_tensor = sess.run([image])
#     # print(image_tensor)
#
#     summary = sess.run([all_summaries])
#     writer.add_summary(summary)
#     writer.close()
#
#     # Finish off the filename queue coordinator.
#     coord.request_stop()
#     coord.join(threads)

import io
import matplotlib.pyplot as plt
import tensorflow as tf


def gen_plot():
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot([1, 2])
    plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


# Prepare the plot
plot_buf = gen_plot()

# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

# Add the batch dimension
image = tf.expand_dims(image, 0)

# Add image summary
summary_op = tf.summary.image("plot", image)

# Session
with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./logs/foo')
    writer.add_summary(summary)
    writer.close()