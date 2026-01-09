from .base import PipelineStep
import pandas as pd
import logging

class RegisterResolution(PipelineStep):
    """
    Step 5: Binding Register Resolution.
    Identifies the binding core within the peptide and labels residues 
    that face the T-cell receptor (TCR) and the MHC anchors.
    """

    def __init__(self):
        super().__init__("Binding Register Resolution")
        # Canonical MHC-II anchor positions (1-indexed)
        self.ANCHORS = {1, 4, 6, 9}

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Identify the binding register and label TCR-facing and Anchor positions.
        """
        logging.info("Step 5: Running Binding Register Resolution")

        if data is None or data.empty:
            logging.warning("Input data is empty.")
            return data

        # Check for required columns from Step 4
        if 'mhc_core_start' not in data.columns:
            logging.warning("MHC core start missing (Step 4 failure?). Resolution skipped.")
            data['tcr_facing_residues'] = None
            data['mhc_anchor_residues'] = None
            return data

        def get_segments(row):
            try:
                peptide = row['peptide_seq']
                core_start = int(row['mhc_core_start'])
                
                tcr_facing_res = []
                tcr_facing_pos = []
                anchors_res = []
                anchors_pos = []
                
                # Standard MHC-II core length is 9
                for i in range(9):
                    peptide_idx = core_start + i
                    if peptide_idx >= len(peptide):
                        continue
                        
                    res = peptide[peptide_idx]
                    pos_in_core = i + 1 # 1-indexed (P1 to P9)
                    
                    if pos_in_core in self.ANCHORS:
                        anchors_res.append(res)
                        anchors_pos.append(peptide_idx)
                    else:
                        tcr_facing_res.append(res)
                        tcr_facing_pos.append(peptide_idx)
                
                return {
                    'tcr_res': "".join(tcr_facing_res),
                    'tcr_pos': tcr_facing_pos,
                    'anchor_res': "".join(anchors_res),
                    'anchor_pos': anchors_pos
                }
            except Exception as e:
                logging.debug(f"Error resolving segments: {e}")
                return None

        # Apply logic
        results = data.apply(get_segments, axis=1)
        
        # Expand results into columns
        data['tcr_facing_residues'] = results.apply(lambda x: x['tcr_res'] if x else None)
        data['tcr_facing_positions'] = results.apply(lambda x: x['tcr_pos'] if x else None)
        data['mhc_anchor_residues'] = results.apply(lambda x: x['anchor_res'] if x else None)
        data['mhc_anchor_positions'] = results.apply(lambda x: x['anchor_pos'] if x else None)

        self.save_results(data, "peptides_register_resolution.csv", **kwargs)

        return data