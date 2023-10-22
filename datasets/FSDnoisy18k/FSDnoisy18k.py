import tensorflow_datasets as tfds
import tensorflow as tf
import os
# import ffmpeg
from pydub import AudioSegment

import glob
from typing import Dict, Any, Generator

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

class FSDNoisy18k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """
        Returns the dataset metadata.

        Returns:
            tfds.core.DatasetInfo: Metadata about the dataset.
        """
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Text(),
                }
            ),
            homepage="https://zenodo.org/record/2529934#.Y0cLUYTP19M"
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> Generator[tfds.core.SplitGenerator, None, None]:
        """
        Returns SplitGenerators.

        Args:
            dl_manager (tfds.download.DownloadManager): Download manager for dataset files.

        Yields:
            tfds.core.SplitGenerator: Split generators.
        """
        extracted_dir = dl_manager.download_and_extract("https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip?download=1")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"directory": extracted_dir},
            ),
        ]

    def _generate_examples(self, directory: str) -> Generator[Dict[str, Any], None, None]:
        """
        Yields examples.

        Args:
            directory (str): Path to the directory containing audio files.

        Yields:
            Dict[str, Any]: A dictionary with audio data.
        """
        for filename in glob.glob(os.path.join(directory, "FSDnoisy18k.audio_test", '*.wav')):
            mp3_audio = convert_to_mp3(filename)
            yield filename, {"audio": mp3_audio}
