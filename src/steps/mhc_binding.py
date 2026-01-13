from .base import PipelineStep
import pandas as pd
import epitopepredict as ep
from epitopepredict import base, sequtils
import logging
import os

class MHCBinding(PipelineStep):
    """
    Step 4: MHC Binding Prediction using epitopepredict (TEPITOPEPan).
    Predicts which peptides bind to target MHC-II receptors.
    """
    
    def __init__(self, alleles=None):
        """
        Initialize the MHC Binding step.
        """
        super().__init__("MHC Binding Prediction")
        self.method = 'tepitope'
        self.alleles = alleles if alleles else ["HLA-DRB1*04:01"]
        self.predictor = base.get_predictor(self.method)
        
        # DEBUGGING: Validate Alleles
        try:
            supported = self.predictor.get_alleles()
            invalid = [a for a in self.alleles if a not in supported]
            if invalid:
                logging.warning(f"Alleles {invalid} might not be supported by {self.method}. "
                                f"Supported examples: {supported[:5]}")
        except Exception as e:
            logging.debug(f"Could not validate alleles: {e}")
            
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Run MHC binding prediction on the peptides.
        """
        logging.info(f"Step 4: Running MHC Binding Prediction ({self.method}) for alleles: {self.alleles}")
        
        if data is None or data.empty:
            logging.warning("Input data is empty.")
            return data

        peptides = data['peptide_seq'].tolist()
        
        # epitopepredict expects a list of sequences
        # The predict_sequences method returns a DataFrame with scores
        # Cols: peptide, allele, score, etc.
        
        # Note: TEPITOPEPan covers MHC-II. Allele format "HLA-DRB1*04:01" is standard.
        
        try:
            # Predict
            # epitopepredict returns a DataFrame with MultiIndex or specific structure
            # predict_sequences(sequences, alleles=alleles)
            logging.info(f"Predicting for {len(peptides)} peptides...")
            results = self.predictor.predict_sequences(peptides, alleles=self.alleles)
            
            # The result is typically:
            #          allele1   allele2
            # peptide1  score     score
            # peptide2  score     score
            # OR a long format depending on version. Let's inspect/standardize.
            # Usually it returns a DataFrame with peptide as index and alleles as columns (wide format)
            # OR standard DataFrame.
            
            # Let's ensure we handle the shape.
            # If it's wide format (alleles as columns), we want to convert to long for the detailed output
            
            # Match Columns
            if 'peptide' in results.columns:
                results.rename(columns={'peptide': 'peptide_seq'}, inplace=True)
            
            # epitopepredict uses 'score' as the column name
            if 'score' in results.columns:
                results.rename(columns={'score': 'binding_score'}, inplace=True)

            # Detailed Results Shape
            interaction_cols = [c for c in results.columns if c in self.alleles]
            if interaction_cols: 
                detailed_df = results.melt(id_vars=['peptide_seq', 'name'], 
                                         value_vars=interaction_cols, 
                                         var_name='allele', 
                                         value_name='binding_score')
            else:
                detailed_df = results
            
            # -----------------------------------------------------------------
            # MAPPING BACK TO ORIGINAL DATA
            # -----------------------------------------------------------------
            # epitopepredict puts the index of the sequence in the 'name' column.
            # We use this to map the best score, core, and position back.
            
            if 'name' in detailed_df.columns:
                detailed_df['name'] = pd.to_numeric(detailed_df['name'], errors='coerce')
                
                # Get row with highest binding score for each original index (name)
                idx_best = detailed_df.sort_values('binding_score', ascending=False).drop_duplicates('name')
                idx_best = idx_best.set_index('name')
                
                data['mhc_binding_score'] = data.index.map(idx_best['binding_score'])
                data['best_allele'] = data.index.map(idx_best['allele'])
                if 'core' in idx_best.columns:
                    data['mhc_core'] = data.index.map(idx_best['core'])
                if 'pos' in idx_best.columns:
                    # 'pos' is the offset of the 9-mer core within the 15-mer peptide
                    data['mhc_core_start'] = data.index.map(idx_best['pos'])
            else:
                # Fallback to sequence matching if name is missing
                best_results = detailed_df.sort_values('binding_score', ascending=False).drop_duplicates('peptide_seq')
                best_results = best_results.set_index('peptide_seq')
                
                data['mhc_binding_score'] = data['peptide_seq'].map(best_results['binding_score'])
                data['best_allele'] = data['peptide_seq'].map(best_results.index.to_series().map(lambda x: best_results.loc[x, 'allele'] if 'allele' in best_results.columns else None))
                # simpler best_allele mapping
                if 'allele' in best_results.columns:
                    data['best_allele'] = data['peptide_seq'].map(best_results['allele'])
                
                if 'core' in best_results.columns:
                    data['mhc_core'] = data['peptide_seq'].map(best_results['core'])
                if 'pos' in best_results.columns:
                    data['mhc_core_start'] = data['peptide_seq'].map(best_results['pos'])
            
            self.save_results(data, "mhc_binding.csv", **kwargs)

            return data

        except Exception as e:
            logging.error(f"Failed to run MHC binding prediction: {e}")
            # Do not crash the pipeline, just return data without scores? 
            # Or raise if critical. Let's log and continue with NaN.
            data['mhc_binding_score'] = None
            return data
