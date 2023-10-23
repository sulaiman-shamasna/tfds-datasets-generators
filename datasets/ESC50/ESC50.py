import tensorflow_datasets as tfds
import tensorflow as tf
import os
from pydub import AudioSegment
import json

json_file_path = "datasets\ESC50\class_labels.json"

def load_class_labels_from_json(json_file_path):
    with open(json_file_path, "r") as json_file:
        class_labels = json.load(json_file)
    return class_labels

class_labels = load_class_labels_from_json(json_file_path)

def convert_to_mp3(file_path: str) -> bytes:
    """
    Converts a WAV audio file to MP3 format using pydub.

    Args:
        file_path (str): Path to the input WAV audio file.

    Returns:
        bytes: Binary MP3 audio data.
    """
    audio = AudioSegment.from_wav(file_path)
    mp3_audio = audio.export(format="mp3").read()
    return mp3_audio

class ESC_50(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Text(),
                    "label": tfds.features.ClassLabel(names=class_labels),
                    "metadata": {
                        "version": tfds.features.Text(),
                        "description": tfds.features.Text(),
                    }
                }
            ),
            supervised_keys=("audio", "label"),
            homepage="https://github.com/karolpiczak/ESC-50",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        extracted_dir = dl_manager.download_and_extract(
            "https://github.com/karoldvl/ESC-50/archive/master.zip"
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"directory": extracted_dir},
            ),
        ]

    def _generate_examples(self, directory):
        metadata_path = os.path.join(directory, "ESC-50-master", "meta", "esc50.csv")
        is_first = True
        with tf.io.gfile.GFile(metadata_path) as f:
            for line in f:
                if is_first:
                    is_first = False
                    continue
                line = line.strip()
                key, _, _, category, _, _, _ = line.split(",")
                file_path = os.path.join(
                    directory, "ESC-50-master", "audio", key
                )
                out = convert_to_mp3(file_path)

                metadata = {
                    "version": "1.0.0",
                    "description": "ESC-50 custom dataset with metadata",
                }
                yield key, {
                    "audio": out,
                    "label": category,
                    "metadata": metadata,
                }
