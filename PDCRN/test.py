import os
import tensorflow as tf
from model import get_model
from config import Configuration
import imageio
import time
cfg = Configuration()

input_path = 'ds/test'
save_path = './result'
os.makedirs(save_path, exist_ok=True)
fileNames = sorted(os.listdir(input_path))

model = get_model([None, None, 3])
ckpt = tf.train.Checkpoint(model=model)
ckpt_dir = 'train_ckpts/checkpoint_folder/best'

chkpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
ckpt.restore(chkpt_manager.latest_checkpoint)
_ = model(tf.ones([1, 1024, 2048, 3]))  # dummy input to build the graph

# print the epoch & val accuracy to make sure that the checkpoint is not corrupt
print(f'Checkpoint restored from epoch {ckpt.epoch.numpy()}')
print(f'Best validation score of the model : {ckpt.max_psnr.numpy}')

avg_time = tf.keras.metrics.Mean()
for i in range(len(fileNames)):
    print(fileNames[i])
    input_image = tf.io.read_file(input_path + '/'+ fileNames[i])
    input_image = tf.image.decode_image(input_image, dtype=tf.dtypes.float32)

    time_init = time.time()
    output_batch = model(tf.expand_dims(input_image, 0), training=False)
    output_img = tf.keras.activations.relu(output_batch, max_value=1)[0]
    avg_time(time.time()-time_init)
    output_img = tf.image.convert_image_dtype(output_img, tf.dtypes.uint8)

    img_path = os.path.join(save_path, fileNames[i])
    imageio.imwrite(img_path, output_img.numpy())

print('inference completed')
print(f'runtime/image = {avg_time.result()}s')
