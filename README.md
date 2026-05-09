# Character‑Level RNN for Next‑Character Prediction

A lightweight, educational implementation of a character‑level recurrent neural network (RNN) in PyTorch. The model is trained on a short string (`"hello world"`) and learns to predict the most likely next character given a prefix. It includes an interactive mode for live testing.

## Features

- **Simple RNN Architecture** – Embedding → RNN → Linear, minimal and easy to understand.
- **Next‑Character Prediction** – Given a few starting characters, the model guesses what comes next.
- **Top‑k Predictions** – Shows the top‑3 most likely next characters with confidence scores.
- **Interactive Console Mode** – Enter your own sequences and see predictions in real time.
- **Reproducible** – Fixed random seed for consistent training results.

## Requirements

- Python 3.8+
- PyTorch (`torch`)

Install with:

```bash
pip install torch
```

No GPU is required – the model is tiny and runs entirely on CPU.

## Installation

1. Clone the repository or download `main.py`.
2. Install PyTorch (see above).
3. Run the script.

## Usage

Run the training and interactive prediction:

```bash
python main.py
```

The script will:
1. Train a character‑level RNN on the string `"hello world"` for 500 epochs.
2. Print the loss every 50 epochs.
3. Evaluate the model on several test prefixes (e.g., `"hell"`, `"wor"`).
4. Display the predicted next character with confidence, plus the top‑3 candidates.
5. Enter an interactive loop where you can type your own prefixes (`quit` to exit).

## Example Output

```
Training on: 'hello world'
==================================================
Epoch [50/500], Loss: 1.2345
Epoch [100/500], Loss: 0.7890
...
Training complete!
==================================================

==================================================
Testing the model
==================================================

Input: 'hell' → Predicted: 'o' (Confidence: 85.3%)
  Top predictions:
    'o': 85.3%
    ' ': 12.1%
    'l': 2.6%

Input: 'wor' → Predicted: 'l' (Confidence: 92.7%)
  Top predictions:
    'l': 92.7%
    'd': 5.1%
    'o': 2.2%

...

==================================================
Interactive mode - Enter your own sequences!
(Type 'quit' to exit)
==================================================

Enter sequence: hel
Predicted next character: 'l' (Confidence: 97.1%)

Enter sequence: quit
```

## File Structure

```
.
├── main.py        # Complete training and prediction script
└── README.md      # This file
```

## Customisation

- **Change the training text** – Modify the `text` variable inside `main()` (e.g., `text = "your string here"`).
- **Adjust model size** – Change `hidden_size` or `num_layers` when creating `CharRNN`.
- **Training hyperparameters** – Tweak `epochs` and `learning_rate` in the `train_model()` call.
- **Sequence length limit** – The `max_len` variable controls how many previous characters the model sees; you can increase it for longer dependencies.

## How It Works

1. **Data preparation** – The training string is split into progressively longer sequences (`"h"`, `"he"`, `"hel"`, …) with their corresponding next character as target.
2. **Padding** – Sequences are padded to the same length with a zero index (representing a “null” character).
3. **Model** – An embedding layer converts character indices to vectors. An RNN processes the sequence, and a linear layer maps the final hidden state to vocabulary‑sized logits.
4. **Training** – Cross‑entropy loss is minimised via the Adam optimiser.
5. **Prediction** – A prefix is padded, fed through the model, and softmax probabilities are computed for the next character.

## Limitations

- Trained on a very short string – the model only knows the characters and patterns present in `"hello world"`.
- The RNN architecture is basic; for larger texts consider using LSTMs/GRUs and longer training.
- Input sequences are truncated/padded to a fixed length, so very long prefixes lose earlier context.

## Contributing

Feel free to open issues or pull requests if you'd like to extend the model (e.g., larger datasets, LSTM support, temperature sampling for creative generation).
