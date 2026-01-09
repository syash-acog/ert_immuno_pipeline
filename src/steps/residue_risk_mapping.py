from .base import PipelineStep
import pandas as pd
import numpy as np
import logging
import ast

class ResidueRiskMapping(PipelineStep):
    """
    Step 7: Residue-Level Risk Mapping.
    Aggregates peptide-level immunogenicity scores to individual residues.
    """

    def __init__(self):
        super().__init__("Residue Risk Mapping")

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Maps immunogenicity scores back to protein residues.
        Returns a DataFrame with residue positions and risk scores.
        """
        logging.info("Step 7: Running Residue-Level Risk Mapping")

        if data is None or data.empty:
            logging.warning("Input data is empty.")
            return data

        required_cols = ['peptide_seq', 'start_pos', 'immunogenicity_score', 'tcr_facing_positions', 'protein_seq']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logging.warning(f"Missing columns for mapping: {missing_cols}. Skipping step.")
            return data

        # Get protein sequence and length from the first row (assuming single protein for now)
        # In a multi-protein scenario, we'd group by protein_id
        protein_seq = data['protein_seq'].iloc[0]
        protein_len = len(protein_seq)
        
        # Initialize risk array (0.0 for low risk)
        # Using a dictionary to map position -> max score
        risk_map = {i: 0.0 for i in range(protein_len)}
        
        # Track which peptides contributed to the risk
        peptide_map = {i: [] for i in range(protein_len)}
        start_pos_map = {i: [] for i in range(protein_len)}

        for _, row in data.iterrows():
            score = row['immunogenicity_score']
            
            # Skip if score is None or NaN
            if score is None or pd.isna(score):
                continue
                
            start_pos = int(row['start_pos'])
            
            # Parse tcr_facing_positions if it's a string representation of a list
            tcr_pos_raw = row['tcr_facing_positions']
            tcr_indices = []
            
            if isinstance(tcr_pos_raw, str):
                try:
                    tcr_indices = ast.literal_eval(tcr_pos_raw)
                except:
                    logging.warning(f"Could not parse tcr_facing_positions: {tcr_pos_raw}")
                    continue
            elif isinstance(tcr_pos_raw, list):
                tcr_indices = tcr_pos_raw
            else:
                continue
                
            # If tcr_indices is None or empty, we might want to default to the whole core 
            # or skip. For now, let's skip if we can't identify TCR contacts.
            if not tcr_indices:
                continue

            # Map scores to absolute positions
            for rel_idx in tcr_indices:
                abs_idx = start_pos + rel_idx
                
                if 0 <= abs_idx < protein_len:
                    # MAX AGGREGATION: Take the highest risk score for this residue
                    current_risk = risk_map[abs_idx]
                    if score > current_risk:
                        risk_map[abs_idx] = float(score)
                    
                    peptide_map[abs_idx].append(row['peptide_seq'])
                    start_pos_map[abs_idx].append(start_pos)

        # Create Result DataFrame
        result_rows = []
        for i in range(protein_len):
            result_rows.append({
                'residue_pos': i + 1,  # 1-based index for bio-friendly output
                'amino_acid': protein_seq[i],
                'risk_score': risk_map[i],
                'contributing_peptides_count': len(peptide_map[i]),
                'peptides': sorted(list(set(peptide_map[i]))) if peptide_map[i] else "None",
                'peptide_start_pos': sorted(list(set(start_pos_map[i]))) if start_pos_map[i] else "None"
            })
            
        df = pd.DataFrame(result_rows)
        
        self.save_results(df, "residue_risk.csv", **kwargs)
        
        return data
