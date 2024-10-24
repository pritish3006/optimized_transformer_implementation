import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

class TextPreprocessor:
    """Preprocessor for English-Hindi parallel text data"""
    
    def __init__(self):
        self.stats: Dict[str, Counter] = {
            'en': Counter(),
            'hi': Counter()
        }
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode characters to canonical form"""
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def clean_english(text: str) -> str:
        """Clean English text"""
        # Convert to lowercase
        text = text.lower()
        
        # Standardize whitespace
        text = ' '.join(text.split())
        
        # Standardize punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Standardize numbers
        text = re.sub(r'\d+', '<num>', text)
        
        return text.strip()
    
    @staticmethod
    def clean_hindi(text: str) -> str:
        """Clean Hindi text"""
        # Standardize whitespace
        text = ' '.join(text.split())
        
        # Remove non-Devanagari characters except basic punctuation
        text = re.sub(r'[^\u0900-\u097F\s.,!?-]', '', text)
        
        # Standardize numbers
        text = re.sub(r'\d+', '<num>', text)
        
        return text.strip()
    
    def preprocess_files(self, 
                        en_file: Path, 
                        hi_file: Path, 
                        output_dir: Path) -> Tuple[Path, Path]:
        """
        Preprocess parallel files and save cleaned versions.
        
        Args:
            en_file: Path to English file
            hi_file: Path to Hindi file
            output_dir: Output directory for cleaned files
            
        Returns:
            Tuple of paths to cleaned files (en_clean, hi_clean)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output files
        en_clean = output_dir / f"{en_file.stem}_clean{en_file.suffix}"
        hi_clean = output_dir / f"{hi_file.stem}_clean{hi_file.suffix}"
        
        # Process files line by line
        with open(en_file, 'r', encoding='utf-8') as en_in, \
             open(hi_file, 'r', encoding='utf-8') as hi_in, \
             open(en_clean, 'w', encoding='utf-8') as en_out, \
             open(hi_clean, 'w', encoding='utf-8') as hi_out:
            
            for en_line, hi_line in zip(en_in, hi_in):
                # Normalize and clean
                en_clean_line = self.clean_english(
                    self.normalize_unicode(en_line)
                )
                hi_clean_line = self.clean_hindi(
                    self.normalize_unicode(hi_line)
                )
                
                # Update statistics
                self.stats['en'].update(en_clean_line.split())
                self.stats['hi'].update(hi_clean_line.split())
                
                # Write cleaned lines
                en_out.write(en_clean_line + '\n')
                hi_out.write(hi_clean_line + '\n')
        
        return en_clean, hi_clean
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Get preprocessing statistics"""
        return {
            'english': {
                'unique_tokens': len(self.stats['en']),
                'total_tokens': sum(self.stats['en'].values()),
                'most_common': self.stats['en'].most_common(10)
            },
            'hindi': {
                'unique_tokens': len(self.stats['hi']),
                'total_tokens': sum(self.stats['hi'].values()),
                'most_common': self.stats['hi'].most_common(10)
            }
        }

def preprocess_opus_data(data_dir: str = "Datasets/raw/en-hi", 
                        output_dir: str = "Datasets/processed/en-hi"):
    """Preprocess all Opus dataset files"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    preprocessor = TextPreprocessor()
    processed_files = {}
    
    # Process each split
    for split in ['train', 'dev', 'test']:
        en_file = data_dir / f"opus.en-hi-{split}.en"
        hi_file = data_dir / f"opus.en-hi-{split}.hi"
        
        if en_file.exists() and hi_file.exists():
            split_output_dir = output_dir / split
            en_clean, hi_clean = preprocessor.preprocess_files(
                en_file, hi_file, split_output_dir
            )
            processed_files[split] = (en_clean, hi_clean)
    
    # Get preprocessing statistics
    stats = preprocessor.get_statistics()
    
    return processed_files, stats