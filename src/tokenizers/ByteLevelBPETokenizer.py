from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple
import re 

# core tokenizer compoenent inputs
from tokenizers import (
    Tokenizer,                                                                  # base tokenizer class
    models,                                                                     # tokenizer models (type of tokenizer)     
    pre_tokenizers,                                                             # pre tokenizer class - for initial text splitting
    decoders,                                                                   # decoder classes - for converting tokens back to text
    trainers,                                                                   # trainer classes - for training the tokenizer
    processors                                                                  # processor classes - for post processing the tokenized output
)

class ByteLevelBPETokenizer:
    """
    byte-level Byte-Pair Encoding (BPE) tokenizer
    implements BPE tokenizer at the byte level. 
    effective for handling any unicode characters and multi-lingual text data
    also includes training, validation, and testing methods

    Attributes:
        vocab_size (int): tokenizer's maximum vocabulary size
        min_frequency (int): minimum frequency of tokens in the training data
    """

    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        # initialize the base BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE())

        # initialize the byte-level pre-tokenizer
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # initialize the byte-level decoder
        self.tokenizer.decoder = decoders.ByteLevel()

        # initialize the byte-level preprocessor
        self.tokenizer.post_preprocessor = processors.ByteLevel(trim_offsets=False)

        # store the configurations
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.metrics: Dict[str, float] = {}

    def train(self,
            files: Union[str, List[str], Dict[str, List[str]]],
            output_dir: str,
            validation_file: Optional[Union[str, List[str]]] = None) -> None:
        """
        train the BPE tokenizer using the provided files

        Args:
            files: path to the training files. - this can be a single file or a list of files
            output_dir: directory to save the trained tokenizer
            validation_file: (optional) path to the validation file.
        """
        files = [files] if isinstance(files, (str, Path)) else files
        files = [str(Path(f)) for f in files]

        # verify that the files exist
        for file in files:
            if not Path(file).exists():
                raise FileNotFoundError(f"File not found: {file}")

        # train the tokenizer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<bos>",                                                                # beginning of sentence tokens
                            "<pad>",                                                                # padding tokens
                            "<eos>",                                                                # end of sentence tokens
                            "<unk>",                                                                # unknown tokens not in the vocabulary
                            "<en>",                                                                 # english language tokens
                            "<hi>"                                                                  # hindi language tokens
                            ]
        )

        self.tokenizer.train(files, trainer)

        # save the tokenizer
        self.save(output_dir)

        # compute the validation metrics if validation file is provided
        if validation_files:
            validation_files = [validation_files] if isinstance(validation_files, (str, Path)) else validation_files
            validation_files = [str(Path(f)) for f in validation_files]

            for file in validation_files:
                if not Path(file).exists():
                    raise FileNotFoundError(f"validation file not found: {file}")

            self._compute_metrics(validation_files)

    def _compute_metrics(self, files: List[str]) -> None:
        """
        compute the metrics for the tokenizer
        """
        total_tokens = 0
        unknown_tokens = 0
        sequence_lengths = []

        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = self.tokenizer.encode(line.strip())
                    total_tokens = total_tokens + len(tokens.ids)
                    unknown_tokens = unknown_tokens + tokens.ids.count(
                        self.tokenizer.token_to_id("<unk>")
                    )
                    sequence_lengths.append(len(tokens.ids))

        # compute the metrics
        self._metrics = {
            'unknown_token_ratio': unknown_tokens / total_tokens if total_tokens > 0 else 0,
            'average_sequence_length': sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
            'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0
        }

    def encode(self, text: str) -> List[int]:
        """encode txt to token ids"""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """decode token ids to text"""
        return self.tokenizer.decode(token_ids)
    
    def save(self, output_dir: str) -> None:
        """save tokenizer to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(Path(output_dir) / "tokenizer.json"))

    def load(self, tokenizer_path: str) -> None:
        """load tokenizer from disk"""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    @property
    def metrics(self) -> dict:
        """return the current validation metrics"""
        return self._metrics.copy()
    
    