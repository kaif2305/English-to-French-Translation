# Simple Sequence-to-Sequence (Seq2Seq) Machine Translation with PyTorch

This project demonstrates a fundamental Sequence-to-Sequence (Seq2Seq) model with an Encoder-Decoder architecture for machine translation using PyTorch. The example focuses on a very small dataset of English-to-French sentences to illustrate the core concepts of sequence processing, vocabulary building, padding, and the training of a basic Seq2Seq model with teacher forcing.

## Project Overview

The Python script implements the following key components:

1.  **Dataset and Preprocessing**: Defines a tiny set of English-French sentence pairs. It includes functions for:
    * Building vocabulary for both source (English) and target (French) languages.
    * Tokenizing sentences, adding special tokens (`<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`), and padding sequences to a fixed maximum length.
2.  **Custom PyTorch Dataset**: Creates a `TranslationDataset` to handle the tokenized source and target sentence pairs, making it compatible with PyTorch's `DataLoader`.
3.  **Encoder Model**: An `Encoder` network built with `nn.LSTM` that processes the input (English) sequence and produces a context vector (the final hidden and cell states) representing the input's meaning.
4.  **Decoder Model**: A `Decoder` network also built with `nn.LSTM` that takes the context vector from the encoder and generates the output (French) sequence word by word.
5.  **Seq2Seq Model**: The overarching `Seq2Seq` model orchestrates the interaction between the Encoder and Decoder, implementing the training process with "teacher forcing."
6.  **Training Loop**: Defines a training function to iteratively train the Seq2Seq model using `Adam` optimizer and `CrossEntropyLoss`.
7.  **Translation Function**: Implements a function to translate a new English sentence into French using the trained model.

## Architecture: Encoder-Decoder with LSTM

The core of this machine translation system is the Encoder-Decoder architecture, typically implemented with Recurrent Neural Networks (RNNs), or more commonly, LSTMs or GRUs due to their ability to handle long-term dependencies.

### Encoder

* **Input**: A sequence of numerical tokens representing an English sentence.
* **Layers**:
    * `nn.Embedding`: Converts input word tokens (integers) into dense vector representations.
    * `nn.LSTM`: Processes the embedded input sequence. The LSTM reads the entire input sequence, and its final hidden state and cell state are used as the "context" or "thought" vector that summarizes the input sentence.
* **Output**: The final hidden state and cell state of the LSTM. These states encapsulate the information from the entire input sequence.

### Decoder

* **Input**:
    * The context vector (initial hidden and cell states) from the Encoder.
    * The previous predicted (or true) token in the target sequence.
* **Layers**:
    * `nn.Embedding`: Converts input word tokens (integers) into dense vector representations.
    * `nn.LSTM`: Takes the embedded input token and the previous hidden/cell states to produce an output for the current time step and update its internal states.
    * `nn.Linear` (Fully Connected Layer): Maps the LSTM's output at each time step to a probability distribution over the entire target language vocabulary.
* **Output**: A probability distribution over the target vocabulary for the next word, along with updated hidden and cell states.

### Seq2Seq Model

The `Seq2Seq` class combines the Encoder and Decoder.

* During training, it passes the source sentence through the encoder to get the initial `hidden` and `cell` states.
* For decoding, it starts with the `<SOS>` token of the target sentence as the initial input to the decoder.
* **Teacher Forcing**: A crucial technique used during training. With a certain `teacher_forcing_ratio` (e.g., 0.5), the model either feeds the *actual* next word from the ground truth target sequence or its *own predicted* word from the previous time step into the decoder for the current time step. This helps the decoder learn faster and more robustly.

## Data Preparation

### Vocabularies

Separate vocabularies are built for English and French sentences. Special tokens are included:
* `<PAD>` (0): Used to fill shorter sequences to a maximum length.
* `<SOS>` (1): Start-of-Sequence token. Indicates the beginning of a sentence.
* `<EOS>` (2): End-of-Sequence token. Indicates the end of a sentence.
* `<UNK>` (3): Unknown token. Used for words not found in the vocabulary.

### Tokenization and Padding

Sentences are converted into sequences of integer tokens. Each sentence is wrapped with `<SOS>` and `<EOS>` tokens, and then padded with `<PAD>` tokens to ensure all sequences have a uniform length (`max_len`).

## Training Process

* **Device**: The model is moved to a GPU (`cuda`) if available, otherwise it runs on the CPU.
* **Optimizer**: `torch.optim.Adam` is used for optimizing the model parameters.
* **Loss Function**: `nn.CrossEntropyLoss` is employed. It's suitable for multi-class classification (predicting the next word from the vocabulary). `ignore_index=french_vocab["<PAD>"]` tells the loss function to ignore padding tokens during loss calculation, as they are not part of the actual sequence content.
* **Training Loop (`train` function)**:
    * Sets the model to training mode (`model.train()`).
    * Iterates over epochs and batches of source and target sentences.
    * Performs a forward pass through the `Seq2Seq` model.
    * Reshapes the output and target tensors to fit the `CrossEntropyLoss` requirements (flattening the sequence and batch dimensions for word prediction).
    * Calculates the loss, performs backpropagation (`loss.backward()`), and updates weights (`optimizer.step()`).

## Translation Inference

The `translate_sentence` function demonstrates how to use the trained model for inference:

1.  **Tokenize Input**: The input English sentence is tokenized and converted to a tensor.
2.  **Encoder Pass**: The tokenized English sentence is passed through the encoder to get the initial `hidden` and `cell` states.
3.  **Decoder Generation**:
    * The decoder starts with the `<SOS>` token as its first input.
    * In a loop, the decoder predicts the next word using its current input and hidden/cell states.
    * The predicted word is then fed back into the decoder as the input for the *next* time step (no teacher forcing here).
    * The process continues until the `<EOS>` token is predicted or the maximum French sentence length is reached.
4.  **Decode Output**: The predicted French token indices are converted back to words to form the translated sentence.

## Setup and Usage

### Prerequisites

* Python 3.x
* `torch`
* `numpy`

You can install these libraries using pip:

```bash
pip install torch numpy