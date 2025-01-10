# Music Generation Using RNN

## Overview

This project explores the use of **Recurrent Neural Networks (RNNs)** for generating musical compositions. Music creation combines creativity, structure, and emotional depth, and advancements in artificial intelligence have enabled outstanding creativity in this domain. RNNs, with their ability to learn and process sequential data, make them a natural choice for composing music, as music fundamentally relies on sequences of notes, chords, and rhythms.

This project focuses on leveraging **Long Short-Term Memory (LSTM)** networks—a specialized type of RNN—to address challenges in music generation and create coherent and engaging compositions.

---

## Key Features

1. **Sequential Data Learning**
   - Processes sequential data such as musical notes and chords.
   - Retains information about prior inputs to understand temporal dependencies.

2. **Long Short-Term Memory (LSTM)**
   - Addresses vanishing gradient problems found in traditional RNNs.
   - Uses memory cells to retain critical data over longer sequences, ensuring coherence over extended passages.

3. **Music Data Encoding and Generation**
   - Encodes musical notes, chords, or structures into a format suitable for the network.
   - Trains the model on patterns and correlations within MIDI files to generate new compositions.

4. **Model Training**
   - Includes hyperparameter tuning for sequence length, batch size, and learning rate.
   - Employs a dataset of encoded musical data to train the network for pattern recognition and composition.

5. **Music Generation**
   - Generates new musical pieces by feeding an initial seed sequence to the trained model and predicting subsequent notes or chords iteratively.

---

## Technologies Used

- **Python**: Programming language for implementation.
- **TensorFlow/Keras**: For building and training RNN and LSTM models.
- **MIDI Libraries**: For processing and encoding musical data (e.g., `music21`, `pretty_midi`).
- **NumPy & Pandas**: For data manipulation and preprocessing.
- **Matplotlib**: For visualizing training metrics and generated music sequences.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/music-generation-rnn.git
   cd music-generation-rnn
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure you have a dataset of MIDI files to train the model. A sample dataset can be downloaded from [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).

---

## Usage

1. **Prepare Data**
   - Place your MIDI files in the `data/` directory.
   - Run the preprocessing script to encode the data into sequences suitable for the RNN:
     ```bash
     python preprocess.py
     ```

2. **Train the Model**
   - Train the RNN model on the preprocessed data:
     ```bash
     python train.py
     ```

3. **Generate Music**
   - Use the trained model to generate a musical composition:
     ```bash
     python generate.py --seed "your_seed_sequence"
     ```

4. **Listen to the Generated Music**
   - Convert the generated sequence back to a MIDI file and play it using any MIDI player.

---

## Future Work

- Experimenting with **Transformer-based models** for music generation.
- Incorporating **style transfer** to mimic the style of specific composers.
- Extending the model to include **lyrics generation** for vocal compositions.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.

---


## Acknowledgments

- Inspired by [DeepMind's WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio).
- MIDI datasets sourced from [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).

