import tensorflow as tf

from model import VAEManager

def main_tests():
    vae_trainer = VAEManager(z_dim=2, beta=0.001, image_dim=(28, 28, 1), seed=1)
    fake_image = tf.zeros((4, 28, 28, 1), dtype=tf.float32)
    encoded = vae_trainer.encode(fake_image)
    assert type(encoded) == dict
    assert encoded['mean'].shape == (4, 2)
    assert encoded['logvar'].shape == (4, 2)
    assert encoded['sample'].shape == (4, 2)
    decoded = vae_trainer.encode_and_decode(fake_image)
    assert decoded.shape == (4, 28, 28, 1)
    generated = vae_trainer.generate()
    assert generated.shape == (1, 28, 28, 1)
    return

if __name__ == "__main__":
    main_tests()