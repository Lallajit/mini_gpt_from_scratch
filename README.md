# Mini GPT From Scratch

A small GPT-style language model implemented using Python and PyTorch. This project demonstrates how a decoder-only Transformer works, including token embeddings, positional embeddings, causal self-attention, multi-head attention, feed-forward networks, and autoregressive text generation.

The model is trained from a text corpus using SentencePiece BPE tokenization and learns to predict the next token in a sequence.

# 1. Project Overview

The objective of this project is to understand and implement the fundamental architecture behind GPT-style language models without relying on a pretrained GPT model.

The model includes:

- SentencePiece BPE tokenization
- Token embeddings
- Positional embeddings
- Scaled dot-product self-attention
- Causal attention masking
- Multi-head attention
- Feed-forward neural networks
- Layer normalization
- Residual connections
- Next-token prediction
- Autoregressive text generation

# 2. Technologies Used

- Python
- PyTorch
- SentencePiece
- Deep Learning
- Natural Language Processing (NLP)
- Transformer Architecture

# 3. Model Architecture

This project implements a decoder-only Transformer architecture.

*Configuration  Value* 
Architecture  Decoder-only Transformer 
Vocabulary size  2,000 *(Select your own choice)*
Embedding dimension  128
Context length  128 tokens 
Transformer blocks  4 
Attention heads per block  4 
Dimension per attention head 32 
Batch size  16 
Learning rate  0.001 
Training loop setting1,400

The model contains four Transformer blocks, each with four attention heads, giving a total of 16 attention heads.

# 4. Project Structure
mini-gpt-from-scratch
main.py
generate.py
transformer_block.py
tokenizer.model
mini_gpt.pth
requirements.txt
.gitignore
README.md


# File descriptions

`main.py`: Trains the SentencePiece tokenizer and the Mini GPT model.
`generate.py`: Loads the trained model and generates text from user prompts.
`transformer_block.py`: Implements self-attention, multi-head attention, feed-forward networks, and Transformer blocks.
`tokenizer.model`: Contains the trained SentencePiece tokenizer.
`mini_gpt.pth`: Contains the trained model weights.

# 5. How the Model Works

Step 1: Tokenization

The training text is converted into token IDs using SentencePiece BPE.

Step 2: Embedding

Each token is converted into a 128-dimensional vector. Positional embeddings are added to represent token positions.

Step 3: Transformer Blocks

The input passes through four Transformer blocks. Each block contains:

- Multi-head self-attention
- Feed-forward neural network
- Layer normalization
- Residual connections

Step 4: Next-Token Prediction

The model predicts the probability distribution of the next token using a linear output layer and cross-entropy loss during training.

Step 5: Text Generation

During generation, the model predicts one token at a time and appends it to the existing sequence. This process continues until the requested number of new tokens is generated.

6. Installation

Clone the repository:

bash
# git clone https://github.com/Lallajit/mini-gpt-from-scratch.git


Navigate to the project directory:

# bash
cd mini-gpt-from-scratch


Create a virtual environment:

# bash
python -m venv myvenv

Activate it on Windows:
myvenv\Scripts\activate


Install the required packages:
pip install -r requirements.txt

# 7. Train the Model

Place your training text file named `corpus.txt` in the project directory.

# 8 Run: python demo.py
for test: python generate.py
Write prompt:
example: human animal
*When I want to exit : press : "exit"*

# 9. Current Limitations

- This is a small experimental language model.
- Generation quality depends on the size and quality of the training corpus.
- The model has a context length of 128 tokens.
- The model has limited vocabulary and model capacity.
- It is not comparable to large pretrained language models such as ChatGPT.
- The model predicts the next token rather than performing reliable question answering.

# 10. Future Improvements

- Increase the size and quality of the training dataset.
- Experiment with larger vocabulary sizes.
- Increase context length and model dimensions.
- Add training and validation loss monitoring.
- Implement model checkpointing.
- Experiment with temperature and top-k sampling.
- Evaluate generated text quality.

# 11. Learning Outcomes

Through this project, I explored the internal working of GPT-style language models and gained practical experience implementing:

- Transformer architecture
- Self-attention mechanisms
- Multi-head attention
- Token and positional embeddings
- Neural network training with PyTorch
- Autoregressive text generation

# From the table of Author(Lallajit)

I developed as a hands-on learning project to understand and implement a small GPT-style language model from scratch.
