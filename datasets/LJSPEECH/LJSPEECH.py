"""LJSpeech Compressed Dataset.

This dataset contains audio data from the LJSpeech dataset in MP3 format.

For more information about LJSpeech, visit: https://keithito.com/LJ-Speech-Dataset/
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
from pydub import AudioSegment

def convert_to_mp3(file_path: str) -> bytes:
    """Converts a WAV audio file to MP3 format using pydub.

    Args:
        file_path (str): Path to the input WAV audio file.

    Returns:
        bytes: Binary MP3 audio data.
    """
    audio = AudioSegment.from_wav(file_path)
    mp3_audio = audio.export(format="mp3").read()
    return mp3_audio

class Ljspeech(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LJSpeech Compressed Dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Text(),
                    "transcript": tfds.features.Text(),
                }
            ),
            supervised_keys=("audio", "transcript"),
            homepage="https://keithito.com/LJ-Speech-Dataset/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        extracted_dir = dl_manager.download_and_extract(
            "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"directory": extracted_dir},
            ),
        ]

    def _generate_examples(self, directory: str):
        metadata_path = os.path.join(directory, "LJSpeech-1.1", "metadata.csv")
        with tf.io.gfile.GFile(metadata_path) as f:
            for line in f:
                line = line.strip()
                key, _, transcript = line.split("|")
                
                file_path = os.path.join(
                    directory, "LJSpeech-1.1", "wavs", "%s.wav" % key
                )
                out = convert_to_mp3(file_path)

                yield key, {
                    "audio": out,
                    "transcript": transcript,
                }
