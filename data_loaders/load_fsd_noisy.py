import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_io as tfio


# This is an example how can the dataset be loaded and ready for further processing. 
# This script will be updated and further implemented later.


def decoding(audio):
    audio["audio"] = tfio.audio.decode_mp3(audio["audio"])
    return audio

ds = tfds.load("fsd_noisy18k", split="train").map(decoding)
for i in ds.take(3):
    print("---->\n", i['audio'])