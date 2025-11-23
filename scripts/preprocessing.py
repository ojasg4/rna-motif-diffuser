"""
Preprocessing script for RNA structure data from Kaggle
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from tqdm import tqdm
import argparse
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RNADataPreprocessor:
    """Preprocess RNA structure data from CSV to NPZ format with parallelization. 
    Meant to be run on the kaggle platform but datapaths can be replaced and run locally."""
    
    def __init__(self, 
                 labels_csv: str = '/kaggle/input/rna-all-data/merged_labels_final.csv',
                 sequences_csv: str = '/kaggle/input/rna-all-data/merged_sequences_final.csv',
                 output_dir: str = './data/preprocessed',
                 contact_threshold: float = 12.0,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 max_seq_len: int = 512,
                 min_seq_len: int = 10):
        self.labels_csv = labels_csv
        self.sequences_csv = sequences_csv
        self.output_dir = Path(output_dir)
        self.contact_threshold = contact_threshold
        self.train_split = train_split
        self.val_split = val_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized preprocessor")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Contact threshold: {self.contact_threshold} Å")
        logger.info(f"Sequence length range: {self.min_seq_len}-{self.max_seq_len}")

    def load_and_filter_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Load CSVs and apply early filtering"""
        logger.info(f"Loading data...")
        
        labels_df = pd.read_csv(self.labels_csv)
        sequences_df = pd.read_csv(self.sequences_csv)
        
        logger.info(f"Loaded {len(labels_df)} coordinate records")
        logger.info(f"Loaded {len(sequences_df)} sequences")

        labels_df['target_id'] = labels_df['ID'].str.rsplit('_', n=1).str[0]
        
        logger.info(f"Extracted target_ids from ID column")
        logger.info(f"Sample IDs: {labels_df['ID'].head(3).tolist()}")
        logger.info(f"Sample target_ids: {labels_df['target_id'].head(3).tolist()}")
        
        sequences_df['seq_len'] = sequences_df['sequence'].str.len()
        
        original_count = len(sequences_df)
        sequences_df = sequences_df[
            (sequences_df['seq_len'] >= self.min_seq_len) & 
            (sequences_df['seq_len'] <= self.max_seq_len)
        ]
        filtered_count = len(sequences_df)
        
        if original_count > filtered_count:
            logger.info(
                f"Filtered {original_count - filtered_count} sequences outside "
                f"length range [{self.min_seq_len}, {self.max_seq_len}]"
            )
            logger.info(f"Remaining: {filtered_count} sequences")

        valid_target_ids = sequences_df['target_id'].unique().tolist()
        
        labels_df = labels_df[labels_df['target_id'].isin(valid_target_ids)]
        
        logger.info(f"Ready to process {len(valid_target_ids)} structures")
        
        return labels_df, sequences_df, valid_target_ids

    def preprocess(self):
        """Main preprocessing function with parallel execution"""
        labels_df, sequences_df, target_ids = self.load_and_filter_data()
        
        np.random.seed(42)
        np.random.shuffle(target_ids)
        
        num_structures = len(target_ids)
        train_end = int(self.train_split * num_structures)
        val_end = int((self.train_split + self.val_split) * num_structures)
        
        splits = {
            'train': target_ids[:train_end],
            'val': target_ids[train_end:val_end],
            'test': target_ids[val_end:],
        }
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(splits['train'])} structures")
        logger.info(f"  Val: {len(splits['val'])} structures")
        logger.info(f"  Test: {len(splits['test'])} structures")
        
        all_tasks = []
        for split_name, ids in splits.items():
            for target_id in ids:
                all_tasks.append({
                    'target_id': target_id,
                    'split': split_name,
                })
        
        num_cpus = max(1, mp.cpu_count() - 1)  # Leave 1 CPU free
        logger.info(f"\nStarting parallel processing on {num_cpus} cores...")
        
        successful = 0
        failed = 0
        
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(
            processes=num_cpus,
            initializer=_worker_init,
            initargs=(self.labels_csv, self.sequences_csv, self.contact_threshold)
        ) as pool:
            results = pool.imap_unordered(_worker_process, all_tasks, chunksize=10)
            
            for result in tqdm(results, total=len(all_tasks), desc="Processing"):
                if result is None:
                    failed += 1
                    continue

                target_id = result['target_id']
                split = result['split']
                data = result['data']
                
                if data is None:
                    failed += 1
                    continue
                
                save_path = self.output_dir / split / f"{target_id}.npz"
                try:
                    np.savez_compressed(save_path, **data)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to save {target_id}: {e}")
                    failed += 1
        
        logger.info("\n" + "="*80)
        logger.info("Preprocessing Complete!")
        logger.info("="*80)
        logger.info(f"Total structures: {num_structures}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {100*successful/num_structures:.1f}%")
        
        self._save_index_files(splits)
        
        logger.info(f"\nOutput saved to: {self.output_dir}")

    def _save_index_files(self, splits: Dict[str, List[str]]):
        """Save index files for each split"""
        for split_name, target_ids in splits.items():
            split_dir = self.output_dir / split_name
            
            saved_files = [f.stem for f in split_dir.glob('*.npz')]
            
            seq_lens = []
            valid_ids = []
            
            for target_id in saved_files:
                npz_path = split_dir / f"{target_id}.npz"
                try:
                    data = np.load(npz_path)
                    seq_len = len(data['seq_ids'])
                    seq_lens.append(seq_len)
                    valid_ids.append(target_id)
                except Exception as e:
                    logger.warning(f"Could not read {target_id} for index: {e}")
            
            if valid_ids:
                index_path = split_dir / 'index.npz'
                np.savez(
                    index_path,
                    target_ids=np.array(valid_ids, dtype=object),
                    seq_lens=np.array(seq_lens, dtype=np.int32)
                )
                logger.info(f"Saved {split_name} index: {len(valid_ids)} structures")

    def verify_preprocessing(self, n_samples: int = 10):
        """Verify that preprocessing worked correctly"""
        logger.info("\n" + "="*80)
        logger.info("Verification")
        logger.info("="*80)
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            npz_files = [f for f in split_dir.glob('*.npz') if f.name != 'index.npz']
            
            if not npz_files:
                logger.warning(f"No files in {split} split!")
                continue
            
            logger.info(f"\n{split.upper()} split: {len(npz_files)} files")
            
            sample_files = np.random.choice(
                npz_files, 
                size=min(n_samples, len(npz_files)),
                replace=False
            )
            
            for npz_file in sample_files:
                try:
                    data = np.load(npz_file)
                    
                    seq_ids = data['seq_ids']
                    coords = data['coords']
                    contact_map = data['contact_map']
                    
                    seq_len = len(seq_ids)
                    
                    assert coords.shape == (seq_len, 3), \
                        f"Coords shape mismatch: {coords.shape}"
                    assert contact_map.shape == (seq_len, seq_len), \
                        f"Contact map shape mismatch: {contact_map.shape}"
                    
                    assert not np.isnan(coords).any(), "Contains NaN coordinates"
                    
                    n_contacts = np.sum(contact_map > 0)
                    
                    logger.info(
                        f"  ✓ {npz_file.name}: len={seq_len}, "
                        f"contacts={n_contacts}, "
                        f"coord_range=[{coords.min():.1f}, {coords.max():.1f}]"
                    )
                    
                except Exception as e:
                    logger.error(f"  ✗ {npz_file.name}: {e}")
        
        logger.info("\n✓ Verification complete!")

# Global variables for workers
_worker_labels_df = None
_worker_sequences_df = None
_worker_contact_threshold = None


def _worker_init(labels_csv: str, sequences_csv: str, contact_threshold: float):
    """Initialize worker process with shared data"""
    global _worker_labels_df, _worker_sequences_df, _worker_contact_threshold
    
    _worker_labels_df = pd.read_csv(labels_csv)
    _worker_sequences_df = pd.read_csv(sequences_csv)
    _worker_contact_threshold = contact_threshold
    
    _worker_labels_df['target_id'] = _worker_labels_df['ID'].str.rsplit('_', n=1).str[0]


def _calc_contact_map(coords: np.ndarray, threshold: float, min_separation: int = 3) -> np.ndarray:
    """
    Calculate contact map with sequence separation filter
    """
    n = len(coords)
    
    distances = pdist(coords, metric='euclidean')
    dist_matrix = squareform(distances)
    
    indices = np.arange(n)
    sep_matrix = np.abs(indices[:, None] - indices[None, :])
    
    contact_map = (
        (dist_matrix < threshold) & 
        (sep_matrix > min_separation)
    ).astype(np.float32)
    
    return contact_map


def _worker_process(task: Dict) -> Optional[Dict]:
    """
    Process a single structure
    """
    global _worker_labels_df, _worker_sequences_df, _worker_contact_threshold
    
    target_id = task['target_id']
    split = task['split']
    
    try:
        seq_row = _worker_sequences_df[_worker_sequences_df['target_id'] == target_id]
        if len(seq_row) == 0:
            return None
        
        sequence = seq_row['sequence'].iloc[0]
        seq_len = len(sequence)
        
        coord_rows = _worker_labels_df[
            _worker_labels_df['target_id'] == target_id
        ].copy()
        
        coord_rows = coord_rows.sort_values('resid')
        
        if len(coord_rows) != seq_len:
            return {
                'target_id': target_id,
                'split': split,
                'data': None
            }
        
        coords = coord_rows[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
        
        if np.isnan(coords).any():
            return {
                'target_id': target_id,
                'split': split,
                'data': None
            }
        
        contact_map = _calc_contact_map(coords, _worker_contact_threshold)
        
        token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        seq_ids = np.array(
            [token_map.get(nuc.upper(), 4) for nuc in sequence],
            dtype=np.int64
        )
        
        return {
            'target_id': target_id,
            'split': split,
            'data': {
                'seq_ids': seq_ids,
                'coords': coords,
                'contact_map': contact_map,
                'sequence': sequence,
                'target_id': target_id
            }
        }
        
    except Exception as e:
        logger.debug(f"Error processing {target_id}: {e}")
        return {
            'target_id': target_id,
            'split': split,
            'data': None
        }

def main():
    parser = argparse.ArgumentParser(
        description="Parallelized RNA Structure Data Preprocessor"
    )
    parser.add_argument('--labels-csv', type=str,
                       default='/kaggle/input/rna-all-data/merged_labels_final.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--sequences-csv', type=str,
                       default='/kaggle/input/rna-all-data/merged_sequences_final.csv',
                       help='Path to sequences CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='./data/preprocessed',
                       help='Output directory')
    parser.add_argument('--contact-threshold', type=float, default=12.0,
                       help='Contact distance threshold (Angstroms)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training split fraction')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split fraction')
    parser.add_argument('--max-seq-len', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--min-seq-len', type=int, default=10,
                       help='Minimum sequence length')
    parser.add_argument('--verify', action='store_true',
                       help='Verify after preprocessing')
    
    args = parser.parse_args()
    
    preprocessor = RNADataPreprocessor(
        labels_csv=args.labels_csv,
        sequences_csv=args.sequences_csv,
        output_dir=args.output_dir,
        contact_threshold=args.contact_threshold,
        train_split=args.train_split,
        val_split=args.val_split,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len
    )
    
    preprocessor.preprocess()
    
    if args.verify:
        preprocessor.verify_preprocessing()


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    main()