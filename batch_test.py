import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='data/images.flist', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--masks', default='data/masks.flist', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--out', default='data/out.flist', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=256, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=256, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='logs/facescape_backup/snap-32000', type=str,
    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print(tf.GraphKeys.GLOBAL_VARIABLES)
    # print(vars_list)
    # input()
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        # print(from_name)
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        # print(var_value)
        # input()
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    with open(args.masks, 'r') as f:
        mlines = f.read().splitlines()
    with open(args.out, 'r') as f:
        olines = f.read().splitlines()
    t = time.time()
    for i in range(len(lines)):
    # for i in range(100):
        # image, mask, out = line.split()
        image = lines[i]
        mask = mlines[i]
        out = olines[i]
        base = os.path.basename(mask)

        image = cv2.imread(image)
        mask = cv2.imread(mask)
        image = cv2.resize(image, (args.image_width, args.image_height))
        mask = cv2.resize(mask, (args.image_width, args.image_height))
        # cv2.imwrite(out, image*(1-mask/255.) + mask)
        # # continue
        # image = np.zeros((128, 256, 3))
        # mask = np.zeros((128, 256, 3))

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        print('Processed: {}'.format(out))
        cv2.imwrite(out, result[0][:, :, ::-1])

    print('Time total: {}'.format(time.time() - t))
