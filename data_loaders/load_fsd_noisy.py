import tensorflow_datasets as tfds

# This is an example how can the dataset be loaded and ready for further processing. 
# This script will be updated and further implemented later.

ds = tfds.load("fsd_noisy18k", split="train")
for i in ds.take(1):
    print("---->\n", type(i['audio']))