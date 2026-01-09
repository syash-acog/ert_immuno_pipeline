from .base import PipelineStep
import pandas as pd
import numpy as np
import logging
import json
import os

class PopulationWeighting(PipelineStep):
    """
    Step 8: Population Weighting & Aggregation.
    Calculates a single protein-level risk score by weighting peptide scores.
    """

    def __init__(self):
        super().__init__("Population Weighting")

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Aggregates peptide scores into a protein-level risk score.
        Inputs: 
            - Peptide DataFrame (from Step 6/7) containing immunogenicity, cleavage, accessibility.
        Outputs:
            - A summary DataFrame (and saves JSON/CSV side-effects).
        """
        logging.info("Step 8: Running Population Weighting")

        if data is None or data.empty:
            logging.warning("Input data is empty.")
            return data

        # Ensure we have the necessary columns
        required_cols = ['immunogenicity_score', 'cleavage_score', 'mean_accessibility']
        for col in required_cols:
            if col not in data.columns:
                logging.warning(f"Missing column '{col}'. Using default weight of 1.0.")
                data[col] = 1.0

        # Allele Frequency:
        # Since we don't have an external allele frequency database linked yet, 
        # and we used a single allele (DRB1*04:01), we assume frequency = 1.0 for this context.
        allele_frequency = 1.0 

        weighted_scores = []
        
        for _, row in data.iterrows():
            # 1. Immunogenicity (Raw Score)
            imm_score = float(row.get('immunogenicity_score'))
            
            # 2. Cleavage Probability (0.0 - 1.0)
            cleavage_prob = float(row.get('cleavage_score')) # Default to 0.5 if missing
            
            # 3. Accessibility Weight
            # SASA is in Angstroms^2. 
            # Normalization: We assume >200 A^2 is fully accessible (weight 1.0).
            sasa = float(row.get('mean_accessibility'))
            if sasa < 0: sasa = 0 # Handle -10.0 sentinels if any
            accessibility_weight = min(sasa / 200.0, 1.0)
            
            # Combine
            # Risk = Score * Frequency * Cleavage * Accessibility
            risk = imm_score * allele_frequency * cleavage_prob * accessibility_weight
            weighted_scores.append(risk)

        # Aggregate
        total_protein_risk = sum(weighted_scores)
        avg_risk = np.mean(weighted_scores) if weighted_scores else 0.0
        
        logging.info(f"Total Protein Risk Score: {total_protein_risk:.4f}")

        # Create Summary
        summary = {
            "protein_id": data['protein_id'].iloc[0] if 'protein_id' in data else 'Unknown',
            "protein_seq": data['protein_seq'].iloc[0] if 'protein_seq' in data else 'Unknown',
            "protein_risk_score": total_protein_risk,
            "average_peptide_risk": avg_risk,
        }
        
        # Save as CSV (User Request)
        summary_df = pd.DataFrame([summary])
        self.save_results(summary_df, "protein_risk.csv", **kwargs)

        return summary_df
