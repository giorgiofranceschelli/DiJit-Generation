from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

def _scatter_plot(vae, example_images, example_labels, path='./images/'):
    z_points = vae.encode(np.asarray(example_images))['sample']
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(1, 1, 1)
    plot_1 = ax.scatter(z_points[:, 0], z_points[:, 1], cmap='rainbow', c= example_labels, alpha=0.5, s=2)
    plt.colorbar(plot_1)
    plt.savefig(path+'scatter', format='png')
        
        
def _reconstruction_plot(vae, real_images, n_to_show=10, path='./images/'):
    rec_images = vae.encode_and_decode(np.asarray(real_images))
    fig, axarr = plt.subplots(n_to_show, 2, figsize=(15, 18))
    for i in range(0, n_to_show):
        axarr[i][0].imshow(real_images[i], cmap='gray')
        axarr[i][0].set_axis_off()
        axarr[i][1].imshow(rec_images[i], cmap='gray')
        axarr[i][1].set_axis_off()
    axarr[0][0].set_title('REAL')
    axarr[0][1].set_title('REC')
    plt.savefig(path+'reconstruction', format='png')
    
    
def _generation_plot(vae, n_to_show=10, path='./images/'):
    fig, axarr = plt.subplots(n_to_show, 2, figsize=(15, 18))
    for i in range(0, n_to_show):
        gen_image = vae.generate()
        axarr[i][0].imshow(gen_image[0], cmap='gray')
        axarr[i][0].set_axis_off()
        gen_image = vae.generate()
        axarr[i][1].imshow(gen_image[0], cmap='gray')
        axarr[i][1].set_axis_off()
    plt.savefig(path+'generation', format='png')
    
    
def make_plots(vae, path='./images/', scatter_to_show=5000, image_to_show=10):
    """    Create and save plots to verify the efficacy of vae training.
    
    Parameters
    ----------
    vae: VAEManager
        The VAE model to be tested.
    path: str, optional
        The path in which storing the images (default './images/').
    scatter_to_show: int, optional
        How many points to scatter when plotting the different classes in a 2d space (default 5000).
    image_to_show: int, optional
        How many images to print when reconstructing or generating from model (default 10).
    """
    def normalize_img(ds):
        return tf.cast(ds['image'], tf.float32) / 255., ds['label']
    if not os.path.exists(path):
        os.makedirs(path)
    (_, ds_test), _ = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, with_info=True)
    ds_test = ds_test.map(normalize_img)
    x_test = list(map(lambda x: x[0].numpy(), ds_test))
    y_test = list(map(lambda x: x[1].numpy(), ds_test))
    example_idx = np.random.choice(range(len(x_test)), scatter_to_show)
    example_images = [x_test[i] for i in example_idx]
    example_labels = [y_test[i] for i in example_idx]
    _scatter_plot(vae, example_images, example_labels, path=path)
    example_idx = np.random.choice(range(len(x_test)), image_to_show)
    real_images = [x_test[i] for i in example_idx]
    _reconstruction_plot(vae, real_images, n_to_show=image_to_show, path=path)
    _generation_plot(vae, n_to_show=image_to_show, path=path)