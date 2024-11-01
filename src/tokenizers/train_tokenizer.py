from pathlib import Path
import os
import numpy as np
import json
from datetime import datetime
import time
from typing import Dict, Any, List, Tuple
from tqdm import tqdm 
from ByteLevelBPETokenizer import ByteLevelBPETokenizer
from TextPreProcessor import TextPreprocessor
import wandb
from dataclasses import dataclass

@dataclass
class HPRange:
    """Hyperparameter range specification"""
    min: int 
    max: int
    step: int = None                                                    # for grid search
    n_samples: int = None                                               # for random search 
    log_scale: bool = False                                             # for log scale search - for parameters that should be samples on a log scale

    def sample(self, n_samples: int = None) -> np.ndarray:
        """sample n_samples from the hyperparameter range"""
        if n_samples is None:
            n_samples = self.n_samples

        if self.log_scale:
            # log uniform sampling
            log_min = np.log(self.min)
            log_max = np.log(self.max)
            samples = np.exp(np.random.uniform(log_min, log_max, n_samples))
        else:
            samples = np.random.uniform(self.min, self.max + 1, n_samples)

        # return the rounded up to the nearest integer values for discrete parameters
        return samples.astype(int)

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    patience: int = 5
    min_improvement: float = 0.01
    min_trials: int = 5

class TokenizerHPTuner:
    """Hyperparameter tuning for the ByteLevelBPETokenizer"""

    def __init__(self,
                 data_dir: str = "/Datasets/raw/en-hi",
                 output_dir: str = "/Datasets/processed/en-hi",
                 tokenizer_dir: str = "/tokenizer",
                 n_trials: int = 10,
                 project_name: str = "tokenizer_tuning",
                 early_stopping_config: EarlyStoppingConfig = EarlyStoppingConfig()
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.n_trials = n_trials
        self.project_name = project_name
        self.early_stopping_config = early_stopping_config

        # define the hyperparameter search space
        self.hp_ranges = {
            'vocab_size': HPRange(
                min=20000,
                max=50000,
                n_samples=n_trials,
                log_scale=True
               ),
            'min_frequency': HPRange(
                min=1,
                max=10,
                n_samples=n_trials,
                log_scale=False
            )
        }

        # create directories
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)

        # store and track the best results
        self.best_results = {
            'params': None,
            'metrics': None,
            'score': float('inf')
        }        

        # Initialize Weights & Biases
        wandb.init(project=self.project_name)

        # Initialize trial history
        self.trial_history = []

        # Initialize early stopping variables
        self.trials_without_improvement = 0
        self.best_score = float('inf')

    def sample_hyperparameters(self) -> List[Dict[str, int]]:
        """sample hyperparameter combinations from the search space for trials"""
        vocab_sizes = self.hp_ranges['vocab_size'].sample()
        min_frequencies = self.hp_ranges['min_frequency'].sample()
        
        # create combinations of hyperparameters
        combinations = []
        for vocab_size, min_freq in zip(vocab_sizes, min_frequencies):
            combinations.append({
                'vocab_size': int(vocab_size),
                'min_frequency': int(min_freq)
            })

        return combinations
    
    def evaluate_tokenizer(self,
                           tokenizer: ByteLevelBPETokenizer,
                           val_files: Tuple[str, str]
    ) -> Dict[str, float]:
        """evaluate the tokenizer performance on validation files"""
        en_file, hi_file = val_files
        unknown_count = 0
        total_tokens = 0
        sequence_lengths = []
        vocab_size = len(tokenizer.vocab)

        # process validation files
        with open(en_file, 'r', encoding='utf-8') as f_en, open(hi_file, 'r', encoding='utf-8') as f_hi:

            for en_line, hi_line in zip(f_en, f_hi):
                # tokenize both languages
                en_tokens = tokenizer.encode(en_line.strip())
                hi_tokens = tokenizer.encode(hi_line.strip())

                # count unknown tokens
                unknown_count = unknown_count + en_tokens.count(tokenizer.token_to_id('<unk>'))
                unknown_count = unknown_count + hi_tokens.count(tokenizer.token_to_id('<unk>'))

                # count total tokens
                total_tokens = total_tokens + len(en_tokens) + len(hi_tokens)

                # record the sequence lengths
                sequence_lengths.extend([len(en_tokens), len(hi_tokens)])

        return {
            'unknown_token_ratio': unknown_count / total_tokens,
            'average_sequence_length': float(np.mean(sequence_lengths)),
            'max_sequence_length': int(max(sequence_lengths)),
            'vocabulary_utilization': vocab_size / tokenizer.vocab_size
        }
                
    def _train_tokenizer(self, params: Dict[str, int], train_files: Tuple[Path, Path], temp_dir: Path) -> ByteLevelBPETokenizer:
        """Train tokenizer with given parameters"""
        tokenizer = ByteLevelBPETokenizer(
            vocab_size=params['vocab_size'],
            min_frequency=params['min_frequency']
        )
        
        tokenizer.train(
            files=[str(f) for f in train_files],
            output_dir=str(temp_dir)
        )
        return tokenizer

    def tune(self) -> Dict:
        """Perform hyperparameter tuning using random search"""
        trial_start_time = time.time()
        train_files = (
            self.data_dir / 'opus.en-hi-train.en',
            self.data_dir / 'opus.en-hi-train.hi'
        )

        val_files = (
            self.data_dir / 'opus.en-hi-dev.en',
            self.data_dir / 'opus.en-hi-dev.hi'
        )

        # sample hyperparameter combinations
        combinations = self.sample_hyperparameters()

        # try combinations
        for trial, params in enumerate(tqdm(combinations, desc="Tuning hyperparameters")):
            # Step 1: Train
            temp_dir = self.tokenizer_dir / f"temp_{trial}"
            tokenizer = self._train_tokenizer(params, train_files, temp_dir)
            
            # Step 2: Evaluate using existing method
            metrics = self.evaluate_tokenizer(tokenizer, val_files)

            # caluclate the score (lower is better)
            score = (
                metrics['unknown_token_ratio'] * 0.4 + 
                (1 - metrics['vocabulary_utilization']) * 0.3 + 
                (metrics['average_sequence_length'] / 512) * 0.3
            )

            # calculate the trial duration
            trial_duration = time.time() - trial_start_time

            # log trial results
            self._log_trial_results(trial, params, metrics, score, trial_duration)

            # update the best result if the current score is better
            if score < self.best_results['score']:
                improvement = self.best_results['score'] - score
                self.best_results = {
                    'params': params,
                    'metrics': metrics,
                    'score': score,
                    'trial': trial,
                    'time': datetime.now().isoformat()
                }

                # Reset early stopping counter
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1

            # Early stopping check
            if self._check_early_stopping(trial, score):
                print(f"Early stopping triggered after {trial + 1} trials.")
                break

            # clean up the temporary directory
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
        
        # save final results
        final_results = {
            'best_results': self.best_results,
            'hp_ranges_used': {
                'vocab_size': {'min': self.hp_ranges['vocab_size'].min, 
                                'max': self.hp_ranges['vocab_size'].max,
                },
                'min_frequency': {'min': self.hp_ranges['min_frequency'].min, 
                                'max': self.hp_ranges['min_frequency'].max,
                }
            },
            'n_trials_completed': len(self.trial_history)
        }

        # Log final results to W&B
        wandb.log({"final_results": final_results})

        # Generate and log visualizations to W&B
        self._log_visualizations()

        return self.best_results['params']

    def _log_trial_results(self, trial, params, metrics, score, duration):
        """Log trial results to W&B and update trial history"""
        wandb.log({
            "trial": trial,
            "vocab_size": params['vocab_size'],
            "min_frequency": params['min_frequency'],
            "unknown_token_ratio": metrics['unknown_token_ratio'],
            "vocabulary_utilization": metrics['vocabulary_utilization'],
            "average_sequence_length": metrics['average_sequence_length'],
            "score": score,
            "duration": duration
        })

        self.trial_history.append({
            "trial": trial,
            "params": params,
            "metrics": metrics,
            "score": score,
            "duration": duration
        })

    def _check_early_stopping(self, trial, score):
        """Check if early stopping criteria are met"""
        if trial < self.early_stopping_config.min_trials:
            return False

        if score < self.best_score - self.early_stopping_config.min_improvement:
            self.best_score = score
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        return self.trials_without_improvement >= self.early_stopping_config.patience

    def _log_visualizations(self):
        """Generate and log visualizations to W&B"""
        # Parameter importance plot
        wandb.log({"parameter_importance": wandb.plot.scatter(
            wandb.Table(data=[[t['params']['vocab_size'], t['params']['min_frequency'], t['score']] for t in self.trial_history],
                        columns=["vocab_size", "min_frequency", "score"]),
            x="vocab_size",
            y="score",
            color="min_frequency"
        )})

        # Score improvement trajectory
        wandb.log({"score_trajectory": wandb.plot.line(
            wandb.Table(data=[[t['trial'], t['score']] for t in self.trial_history],
                        columns=["trial", "score"]),
            x="trial",
            y="score"
        )})

        # Trial duration tracking
        wandb.log({"trial_duration": wandb.plot.line(
            wandb.Table(data=[[t['trial'], t['duration']] for t in self.trial_history],
                        columns=["trial", "duration"]),
            x="trial",
            y="duration"
        )})

        # Best parameter evolution
        best_vocab_sizes = []
        best_min_frequencies = []
        best_score = float('inf')

        for trial in self.trial_history:
            if trial['score'] < best_score:
                best_score = trial['score']
                best_vocab_sizes.append(trial['params']['vocab_size'])
                best_min_frequencies.append(trial['params']['min_frequency'])
            else:
                best_vocab_sizes.append(best_vocab_sizes[-1])
                best_min_frequencies.append(best_min_frequencies[-1])

        wandb.log({"best_parameter_evolution": wandb.plot.line(
            wandb.Table(data=[[t['trial'], vs, mf] for t, vs, mf in zip(self.trial_history, best_vocab_sizes, best_min_frequencies)],
                        columns=["trial", "best_vocab_size", "best_min_frequency"]),
            x="trial",
            y=["best_vocab_size", "best_min_frequency"]
        )})


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    early_stopping_config = EarlyStoppingConfig(patience=5, min_improvement=0.01, min_trials=5)
    tuner = TokenizerHPTuner(
        data_dir=os.path.join(project_root, "Datasets", "raw", "en-hi"),
        output_dir=os.path.join(project_root, "Datasets", "processed", "en-hi"),
        tokenizer_dir=os.path.join(project_root, "tokenizers", "tokenizer"),
        n_trials=10,
        project_name="tokenizer_tuning",
        early_stopping_config=early_stopping_config
    )
    
    # debug prints to verify the paths
    print(f"Project Root: {project_root}")
    print(f"Expected Input Data directory: {tuner.data_dir}")
    print(f"Expected Output Data directory: {tuner.output_dir}")
    print(f" Tokenizer directory: {tuner.tokenizer_dir}")

    best_params = tuner.tune()
    print("\nTuning complete!")
    print(f"Best parameters found (score: {tuner.best_results['score']:.4f}):")
    print(json.dumps(best_params, indent=2))