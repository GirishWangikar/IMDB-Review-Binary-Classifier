# IMDB-Review-Binary-Classifier

Dataset
The IMDB Reviews dataset is hosted on TensorFlow Datasets and contains 100,000 movie reviews. The dataset is split into 'train', 'test', and 'unsupervised' sets. For this project, we use only the 'train' and 'test' sets to train and evaluate the model.

Preprocessing
The movie review text undergoes tokenization and padding to ensure uniform length. TensorFlow's Tokenizer and pad_sequences functions are used for this purpose.

Model Architecture
The model architecture consists of the following layers:

Embedding layer: Represents each word in the vocabulary with trainable word vectors.
Flatten layer: Flattens the output of the Embedding layer.
Dense layer: Applies ReLU activation function to the flattened output.
Output layer: A single neuron with a sigmoid activation function for binary classification.
Training
The model is trained using binary cross-entropy loss and the Adam optimizer. The training process is executed for a specified number of epochs.

Evaluation
The model's performance is evaluated on the test set to measure its accuracy in predicting sentiment.

Visualizing Word Embeddings
The trained word embeddings are visualized using the TensorFlow Embedding Projector. The word embeddings are saved to files vecs.tsv and meta.tsv, which contain the vector weights of each word in the vocabulary and the corresponding words, respectively.

Usage
To run this project, follow these steps:

Download the IMDB Reviews dataset using the provided code snippet.
Preprocess the text data to tokenize and pad the movie reviews.
Build the binary classifier model with the specified architecture.
Train the model on the preprocessed data.
Evaluate the model's performance on the test set.
Visualize the word embeddings using the TensorFlow Embedding Projector.
