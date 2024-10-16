# Transformer Implementation

This project implements the Transformer model as described in the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. The implementation is built from scratch using PyTorch, with the goal of training and optimizing it for various tasks.

## Project Overview

- **Purpose**: Implement a Transformer from scratch, train it, and optimize it for various tasks.
- **Architecture**: Faithful to the original paper, without specific improvements or additional features.
- **Current Focus**: Tuning for Apple Silicon and Metal Performance Shaders (transitioning to CUDA-based approach in the future).
- **Status**: In development and currently being trained.
- **Dataset**: Using OPUS-100 en-hi dataset for initial experimentation. OPUS-100 is a multilingual dataset derived from the OPUS collection, containing sentence pairs for 100 language pairs. The en-hi subset consists of English-Hindi parallel sentences, providing a diverse range of text for training and evaluating machine translation models.

## Main Components

The main components of the project can be found in the `src` directory:

- `models/`: Contains implementations of the Transformer's core components (encoder, decoder, attention mechanisms, etc.)
- `data/`: Includes data loading and processing utilities.
- `utils/`: Houses helper functions and utilities.

## Setup

1. Create conda environment:
   `conda env create -f conda-env.yml`

2. Activate environment:
   `conda activate transformer_env`

3. Install the project in editable mode:
   `pip install -e .`

## Usage

While the model is still in development, users can train it for their own purposes. Here are basic instructions to get started:

1. Prepare your dataset in a format compatible with the `TextDataLoader` in `src/data/text_dataloader.py`.
2. Initialize the Transformer model using the `Transformer` class in `src/models/transformer.py`.
3. Set up your training loop, utilizing the components by populating the `train.py` file or creating a Jupyter notebook for the training loop.

Detailed usage examples and a training script will be provided in future updates.

## Contributing

Contributions are welcome! We appreciate any ideas and collaboration on the experimentation. Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper

## Future Work

- Transition to CUDA-based approach for broader GPU support
- Optimize performance and expand to more tasks and datasets
- Provide comprehensive documentation and usage examples
