import os
import sys
import subprocess
import pandas as pd
import freesasa
from Bio import PDB
from .base import PipelineStep

class StructuralAccessibility(PipelineStep):
    """
    Step 3: Calculates structural accessibility (SASA) of peptides.
    Requires a PDB file. If not provided, attempts to generate one using ColabFold.
    """

    def __init__(self):
        super().__init__("Structural Accessibility")

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if data is None or data.empty:
            return data

        pdb_path = kwargs.get('pdb_path')
        output_dir = kwargs.get('output_dir', 'data/output')
        
        # 1. Structure Acquisition
        structure_file = None
        
        if pdb_path and os.path.exists(pdb_path):
            print(f"Using provided PDB file: {pdb_path}")
            structure_file = pdb_path
        else:
            print("No PDB file provided or file not found.")
            # We need the input FASTA path to run ColabFold
            fasta_path = kwargs.get('fasta_path')
            
            if fasta_path and os.path.exists(fasta_path):
                print("Attempting to generate structure using ColabFold (requires GPU)...")
                generated_pdb = self._run_colabfold(fasta_path, output_dir)
                if generated_pdb:
                    structure_file = generated_pdb
            else:
                print("Warning: Missing FASTA path for structure generation.")

        # 2. SASA Calculation
        if structure_file:
            print(f"Calculating SASA using: {structure_file}")
            try:
                # ---------------------------------------------------------
                # EXPLANATION: How we calculate the Accessibility
                # ---------------------------------------------------------

                # 1. Load the Structure
                # We tell the FreeSASA library to read the PDB file.
                # It parses the 3D coordinates of every atom.
                structure_sasa = freesasa.Structure(structure_file)
                
                # 2. Calculate (SASA)
                # This function simulates rolling a probe (like a water molecule)
                # over the protein. It calculates how much surface area of each atom is "touchable" by the water.
                result = freesasa.calc(structure_sasa)
                
                # 3. Get Per-Residue Area
                # We don't care about atoms (too detailed). We want RESIDUES (amino acids).
                # This function groups the area by residue.
                # Format: residue_areas['ChainID']['ResidueNumber'] = <AreaObject>
                residue_areas = result.residueAreas()
                
                # 4. Identify the Chain
                # PDB files often have multiple chains (A, B, C...).
                # Since we likely just have one protein, we grab the first available chain.
                available_chains = list(residue_areas.keys())
                target_chain = available_chains[0] if available_chains else 'A'
                print(f"Using Chain {target_chain} for SASA mapping.")
                
                # Get the map of ResidueNumber -> Area for that chain
                chain_areas = residue_areas.get(target_chain, {})
                
                # 5. Score Each Peptide
                accessibility_scores = []
                
                for idx, row in data.iterrows():
                    peptide = row['peptide_seq'] 
                    protein_seq = row['protein_seq']
                    
                    if not peptide or not protein_seq:
                        accessibility_scores.append(None)
                        continue
                        
                    # Find where this peptide lives in the full protein
                    # start_pos is 0-based index (Python style)
                    start_pos = protein_seq.find(peptide)
                    if start_pos == -1:
                        accessibility_scores.append(None)
                        continue
                    
                    pep_len = len(peptide)
                    
                    # -----------------------------------------------------
                    # CRITICAL: Mapping Python Index to PDB Index
                    # -----------------------------------------------------
                    # Python starts counting at 0. PDB files start counting at 1.
                    # So, Python Index 0 = PDB Residue 1.
                    # We create a range of numbers representing the PDB residue IDs.
                    pdb_indices = range(start_pos + 1, start_pos + 1 + pep_len)
                    
                    # Collect SASA values for every letter in this peptide
                    sasa_values = []
                    for res_idx in pdb_indices:
                        # Extract the area for this specific residue number
                        # We convert res_idx to string because the library uses string keys
                        val = chain_areas.get(str(res_idx))
                        
                        # Fallback for integer keys just in case
                        if val is None:
                            val = chain_areas.get(res_idx) 
                            
                        # If we found data, add the 'total' area of that residue to our list
                        if val is not None:
                            sasa_values.append(val.total) 
                        else:
                            # If val is None, it means this residue is MISSING in the PDB file.
                            # (Common in N-terminal or C-terminal tails that are disordered)
                            pass
                            
                    # Calculate the Average (Mean) Accessibility for this peptide
                    if sasa_values:
                        mean_sasa = sum(sasa_values) / len(sasa_values)
                        accessibility_scores.append(mean_sasa)
                    else:
                        # If no residues were found (e.g., peptide is in a missing tail)
                        # We assign 0.0 (treated as buried/unknown).
                        accessibility_scores.append(0.0)
                        
                data['mean_accessibility'] = accessibility_scores
                
            except Exception as e:
                print(f"Error calculating SASA: {e}")
                data['mean_accessibility'] = None
        else:
            print("Skipping SASA calculation (No PDB).")
            data['mean_accessibility'] = None

        self.save_results(data, "peptides_structural_accessibility.csv", **kwargs)

        return data

    def _run_colabfold(self, fasta_path, output_dir):
        """
        Runs colabfold_batch command.
        """
        struct_dir = os.path.join(output_dir, "structures")
        os.makedirs(struct_dir, exist_ok=True)
        
        # Check if colabfold_batch is in path
        cmd = ["colabfold_batch", fasta_path, struct_dir]
        
        print(f"Running ColabFold: {' '.join(cmd)}")
        try:
            # allowing standard output to show progress
            subprocess.run(cmd, check=True)
            
            # Find the generated PDB
            # ColabFold outputs usually named like <input_basename>_..._rank_1_... .pdb
            # We pick the rank 1 model (best)
            # Simple heuristic: find newest .pdb file in directory that matches base name
            base_name = os.path.splitext(os.path.basename(fasta_path))[0]
            
            candidates = []
            for img in os.listdir(struct_dir):
                if img.endswith(".pdb") and "rank_001" in img and base_name in img:
                    candidates.append(os.path.join(struct_dir, img))
            
            if candidates:
                # Return the one with highest confidence or just the first rank_001
                return candidates[0]
            
            # Fallback: look for any pdb with basename
            for img in os.listdir(struct_dir):
                if img.endswith(".pdb") and base_name in img:
                     return os.path.join(struct_dir, img)

        except FileNotFoundError:
            print("Error: 'colabfold_batch' command not found. Is LocalColabFold installed?")
        except subprocess.CalledProcessError as e:
            print(f"ColabFold failed: {e}")
            
        return None
