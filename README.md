# Capuchin Bird call Detection and Counting
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00)
![Keras](https://img.shields.io/badge/Keras-2.12%2B-D00000)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243)
![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-FF3399)

This project develops a deep learning model to detect and count Capuchin bird calls in audio recordings, using the Z by HP Unlocked Challenge 3 - Signal Processing dataset from Kaggle. The solution is implemented using TensorFlow, with Short-Time Fourier Transform (STFT) spectrograms as input features and a custom Convolutional Neural Network (CNN) for binary classification (Capuchin vs. non-Capuchin). The model is trained on short audio clips and applied to longer forest recordings to count bird calls via a sliding window approach.
## How the Counting Funtion work ?
### Overview
The funtion analyzes audio files to detect and count bird vocalizations using a machine learning model. It processes audio in sliding time windows and returns the number of segments containing bird calls.First it loads the audio file at 16Hz sample rate in mono format,Divides it into three segments each having 3-seconds with 1-second stride(2-second overlap) then convert audio segment into a Spectrogram using STFT,use the custome CNN model to classify wheatehr the spectrogram contains bird or not if it contains then increase the count (if it exceed the threshold).Output total count of audio segments contains bird calls.
## Audio Preprocessing Steps
- **Pad the audio files to a fixed length**:Adjust each audio clip to a fixed duration of 3 seconds,to ensure all the inputs have the same lengths.
- **Compute STFT Spectrogram**: Convert time domain audio into a frequency representation, a 2-D representation ( frequency vs time).
- **Convert to Magnitude Spectrogram**: Compute the magnitude of the STFT to determine the amplitude information,which will help to identify bird call patterns.
- **Convert to Log-Scale(dB)**: Convert magnitude spectrogram to decibles,making frequency patterns more distinguishble for the CNN.
- **Normalize to [0,1]**: Standardize  the input features for better CNN training.
- **Add Channel Dimensions**: Reshape the spectrogram by adding channel dimension (513,94) -> (513,94,1), mimiking a single channel image (greyscale).
- **Set Tensor shapes**:  Define the output shapes in the TensorFlow pipeline to prevent shape related errors in batching and training.
- **Data Set Pipeline Creation**: Integrate preprocessed audio into a tf.Dataset pipeline with additional operations (map,shuffle,Batch,Prefetch,Split).
## Parameters
- **n_ff**t: Number of FFT (Fast Fourier Transform) points needed to compute STFT spectrogram.
- **FREQ_BINS**: Number of frequency bins in STFT spectrogram.
- **hop_length**: Number of audio samples between consecutive STFT window, controlling the time resolution of the spectrogram.
