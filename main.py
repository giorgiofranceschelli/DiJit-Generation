import flax
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

from model import VAEManager
from utils import *

flax.config.update('flax_use_orbax_checkpointing', True)

def define_config():
    config = {}
    config['BATCH_SIZE'] = 64
    config['EPOCHS'] = 100
    config['Z_DIM'] = 2
    config['BETA'] = 0.001
    config['LR'] = 0.0001
    config['SEED'] = 1
    return config


def main_training(args):
    
    def normalize_img(ds):
        return tf.cast(ds['image'], tf.float32) / 255., ds['label']
    set_seed(int(args.SEED))
    batch_size = int(args.BATCH_SIZE)
    (dataset, _), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, with_info=True)
    dataset = dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(ds_info.splits['train'].num_examples)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = iter(tfds.as_numpy(dataset))
    num_of_batches = ds_info.splits['train'].num_examples//batch_size
    epochs = int(args.EPOCHS)
    vae_trainer = VAEManager(z_dim=int(args.Z_DIM), beta=float(args.BETA), image_dim=(28, 28, 1), seed=int(args.SEED))
    print("Training starting ...")
    for i in range(epochs):
        rec_loss_epoch = np.zeros((), np.float32)
        reg_loss_epoch = np.zeros((), np.float32)
        for _ in range(num_of_batches):
            batch = next(dataset)
            losses = vae_trainer.train_step(batch[0])
            rec_loss_epoch += losses['rec']
            reg_loss_epoch += losses['reg']
        print("Epoch %d finished with rec loss %.4f and reg loss %.4f" % (i+1, round(rec_loss_epoch/num_of_batches, 4), round(reg_loss_epoch/num_of_batches, 4)))
    vae_trainer.save_model('./models/', name='vae_'+str(epochs)+'epochs_'+str(batch_size)+'bs_'+str(args.Z_DIM)+'zdim')
    print("Training finished.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE on MNIST dataset.')
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()
    main_training(args)
