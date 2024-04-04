import numpy as np
import librosa.display
import os
import matplotlib.pyplot as plt


def create_spectrogram(audio_file: str, image_file: str) -> None:
    """
    Creates a spectrogram from an audio file and saves it to an image file.
    This function uses Librosa to load the audio file, compute its Mel-frequency spectrogram, and save it as an image file.

    Parameters:
        audio_file (str): The path to the audio file.
        image_file (str): The path to the output image file.

    Returns:
        None        
    """
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


    # Mel-frequency spectrogram

    y, sr = librosa.load(audio_file)

    ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    log_ms = librosa.power_to_db(ms, ref=np.max)

    librosa.display.specshow(log_ms)


    fig.savefig(image_file)
    plt.close(fig)


def create_spectrogram_from_audio(input_path: str, output_path: str) -> None:
    """
    Creates spectrogram images for the audio files in the input directory and saves them to the output directory.

    Parameters:
        input_path (str): The path to the directory containing the audio files.
        output_path (str): The path to the directory where the generated images will be saved.

    Returns:
        path where the generated image is saved
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        extension = "."+file.split(".")[-1]
        output_file = os.path.join(output_path, file.replace(extension, '.jpg'))
        create_spectrogram(input_file, output_file)
        
    return output_file


