from .base import PipelineStep
import pandas as pd
import numpy as np
import logging
import os
import subprocess
import sys
import tempfile

class ImmunogenicityScoring(PipelineStep):
    """
    Step 6: Immunogenicity Pattern Scoring using ImmuScope-IM.
    Predicts if a peptide triggers a CD4+ T-cell response.
    """

    def __init__(self, allele: str = "HLA-DRB1*04:01", immuscope_path: str = None):
        super().__init__("Immunogenicity Scoring")
        self.allele_original = allele
        self.allele = self._convert_allele_format(allele)
        
        # Default path to ImmuScope
        if immuscope_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.immuscope_path = os.path.join(base_dir, "tools", "ImmuScope")
        else:
            self.immuscope_path = immuscope_path
        
        # Check if ImmuScope exists
        if not os.path.exists(self.immuscope_path):
            logging.warning(f"ImmuScope not found at {self.immuscope_path}. Scoring will be skipped.")
            self.available = False
        else:
            self.available = True
            # Add ImmuScope to path for imports
            if self.immuscope_path not in sys.path:
                sys.path.insert(0, self.immuscope_path)

    def _convert_allele_format(self, allele: str) -> str:
        """
        Convert standard HLA nomenclature to ImmuScope format.
        HLA-DRB1*04:01 -> DRB1_0401
        """
        # Remove HLA- prefix if present
        if allele.startswith("HLA-"):
            allele = allele[4:]
        
        # Replace * with _ and remove :
        allele = allele.replace("*", "_").replace(":", "")
        
        return allele

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Run immunogenicity scoring on the peptides.
        """
        logging.info("Step 6: Running Immunogenicity Scoring (ImmuScope-IM)")

        if data is None or data.empty:
            logging.warning("Input data is empty.")
            return data

        if not self.available:
            logging.warning("ImmuScope not available. Skipping immunogenicity scoring.")
            data['immunogenicity_score'] = None
            return data

        try:
            # Check for model weights
            model_path = os.path.join(self.immuscope_path, "weights", "IM")
            if not os.path.exists(model_path):
                logging.warning(f"ImmuScope model weights not found at {model_path}.")
                logging.warning("Please download weights from Zenodo and extract to tools/ImmuScope/weights/")
                data['immunogenicity_score'] = None
                return data

            # Import ImmuScope modules
            from ImmuScope.utils.data_utils import get_mhc_name_seq, get_peptide_embedding, ACIDS
            from ImmuScope.datasets.datasets import SinInstanceBag
            from ImmuScope.models.trainer_immunogenicity import Trainer
            from ImmuScope.models.ImmuScope import ImmuScope
            import torch
            import h5py
            
            # Load MHC pseudosequences
            mhc_seq_file = os.path.join(self.immuscope_path, "data", "raw", "pseudosequence.2023.dat")
            if not os.path.exists(mhc_seq_file):
                logging.warning(f"MHC pseudosequence file not found at {mhc_seq_file}.")
                data['immunogenicity_score'] = None
                return data
            
            mhc_name_seq = get_mhc_name_seq(mhc_seq_file)
            
            # Check if allele is supported
            if self.allele not in mhc_name_seq:
                logging.warning(f"Allele {self.allele} not found in ImmuScope's MHC database.")
                logging.warning(f"Available alleles (sample): {list(mhc_name_seq.keys())[:5]}")
                data['immunogenicity_score'] = None
                return data

            # Prepare peptides for scoring
            peptides = data['peptide_seq'].tolist()
            
            # Create temporary HDF5 file with peptide data
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                tmp_h5_path = tmp_file.name
            
            try:
                # Convert peptides to ImmuScope format
                peptide_len = 21  # ImmuScope default
                padding_idx = 0
                peptide_pad = 3
                
                peptide_embeddings = get_peptide_embedding(
                    peptides, 
                    peptide_len=peptide_len,
                    padding_idx=padding_idx, 
                    peptide_pad=peptide_pad
                )
                
                # Create HDF5 file
                with h5py.File(tmp_h5_path, 'w') as h5:
                    dt = h5py.string_dtype(encoding='utf-8')
                    mhc_names = [self.allele] * len(peptides)
                    peptide_contexts = [""] * len(peptides)  # No context available
                    labels = [0.0] * len(peptides)  # Dummy labels
                    
                    h5.create_dataset('mhc_names', data=mhc_names, dtype=dt)
                    h5.create_dataset('peptide_embedding', data=peptide_embeddings, dtype=np.int32)
                    h5.create_dataset('peptide_contexts', data=peptide_contexts, dtype=dt)
                    h5.create_dataset('labels', data=labels, dtype=np.float32)
                
                # Load model and run inference
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logging.info(f"Using device: {device}")
                
                # Find model files
                model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
                if not model_files:
                    logging.warning(f"No model files found in {model_path}.")
                    data['immunogenicity_score'] = None
                    return data
                
                # Use the first model file (or ensemble if multiple)
                model_file = os.path.join(model_path, model_files[0])
                logging.info(f"Loading model from {model_file}")
                
                # Create MHC embedding dict
                mhc_embedding_dict = {}
                mhc_seq = mhc_name_seq[self.allele]
                mhc_embedding = np.asarray([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])
                mhc_embedding_dict[self.allele] = np.expand_dims(mhc_embedding, axis=0)
                
                # Load data
                from torch.utils.data import DataLoader
                test_loader = DataLoader(
                    SinInstanceBag(tmp_h5_path, mhc_name_seq, indices=None),
                    batch_size=32
                )
                
                # Initialize trainer and predict
                # Model config from configs/ImmuScope-IM.yaml
                model_config = {
                    'emb_size': 16,
                    'conv_size': [9],
                    'conv_num': [64],
                    'conv_off': [3],
                    'dropout': 0.25,
                    'peptide_pad': 3,
                    'bag_size': 10,
                }
                
                trainer = Trainer(
                    ImmuScope, 
                    model_path=model_file, 
                    device=device, 
                    logger=logging.getLogger(),
                    **model_config
                )
                
                # Run prediction
                predictions, _, _ = trainer.predict(test_loader, model_prefix="")
                
                # Map predictions back to data
                data['immunogenicity_score'] = predictions
                
                logging.info(f"Immunogenicity scoring complete. Scores range: [{min(predictions):.3f}, {max(predictions):.3f}]")
                
                # Save intermediate result as requested 
                self.save_results(data, "peptides_immunogenicity_score.csv", **kwargs)
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_h5_path):
                    os.remove(tmp_h5_path)
            
            return data

        except ImportError as e:
            logging.error(f"Failed to import ImmuScope modules: {e}")
            logging.error("Please ensure PyTorch is installed: pip install torch")
            data['immunogenicity_score'] = None
            return data
        except Exception as e:
            logging.error(f"Failed to run immunogenicity scoring: {e}")
            data['immunogenicity_score'] = None
            return data
