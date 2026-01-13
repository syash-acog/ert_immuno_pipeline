import pandas as pd
import sys
import os
import shutil
import subprocess
import tempfile
from .base import PipelineStep

class CleavageLikelihood(PipelineStep):
    """
    Step 2: Predicts cleavage likelihood using NetCleave.
    """

    def __init__(self):
        super().__init__("Cleavage Likelihood (NetCleave)")

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Runs NetCleave on the peptides.
        Generate 7-mer cleavage sites: 4 residues from C-term of peptide + 3 residues flanking in protein.
        """
        if data is None or data.empty:
            print("No peptides to process.")
            return data

        print(f"Processing {len(data)} peptides for cleavage likelihood via NetCleave...")
        
        # Determine paths relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

        potential_paths = [
            "/app/tools/NetCleave/NetCleave.py", # Docker
            os.path.join(project_root, "tools", "NetCleave", "NetCleave.py"), # Local standard
            os.path.join(project_root, "ert_immuno_pipeline", "tools", "NetCleave", "NetCleave.py"), # Nested
            os.path.join(os.getcwd(), "tools", "NetCleave", "NetCleave.py") # Fallback to CWD
        ]
        
        netcleave_path = None
        for path in potential_paths:
            if os.path.exists(path):
                netcleave_path = path
                break
        
        if not netcleave_path:
            print(f"Warning: NetCleave not found. Searched: {potential_paths}")
            raise ValueError("NetCleave not found. Please install it.")
            
        print(f"Found NetCleave at: {netcleave_path}")

        # Generate 7-mer cleavage sites
        # Data has columns: 'peptide_seq', 'protein_id', 'protein_seq'
        
        cleavage_sites = []
        valid_indices = []
        
        for idx, row in data.iterrows():
            peptide = row['peptide_seq']
            prot_seq = row['protein_seq']
            
            # Find peptide in protein to get flanking regions
            # Note: This finds the FIRST occurrence.
            start_pos = prot_seq.find(peptide)
            if start_pos == -1:
                cleavage_sites.append(None)
                continue
                
            end_pos = start_pos + len(peptide) # Index after last residue
            
            # Check if we have 3 residues after
            if end_pos + 3 <= len(prot_seq):
                # 4 residues from end of peptide + 3 residues from protein
                # Peptide end is at end_pos. Peptide[-4:] is prot_seq[end_pos-4:end_pos]
                # Flanking is prot_seq[end_pos:end_pos+3]
                c_site = prot_seq[end_pos-4:end_pos+3].upper()
                cleavage_sites.append(c_site)
                valid_indices.append(idx)
            else:
                cleavage_sites.append(None) # C-terminus of protein, no cleavage site context
        
        # Filter for NetCleave input
        nets_input = pd.DataFrame({'sequence': [cs for cs in cleavage_sites if cs is not None]})
        
        if nets_input.empty:
            print("No valid cleavage sites generated (peptides might be at C-terminus or not found).")
            data['cleavage_score'] = 0.0
            return data

        # Create a temp file (TSV)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_in:
            input_csv_path = tmp_in.name
            
        try:
            # NetCleave read_data_table uses sep="\t"
            nets_input.to_csv(input_csv_path, sep='\t', index=False)
            
            # Construct command
            cmd = [
                sys.executable,
                netcleave_path,
                "--score_csv", input_csv_path,
                "--mhc_class", "I", 
                "--technique", "mass_spectrometry",
                "--mhc_family", "HLA"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(netcleave_path))
            
            if result.returncode != 0:
                print(f"NetCleave Error: {result.stderr}")
                print(f"NetCleave Stdout: {result.stdout}")
                raise RuntimeError(f"NetCleave failed with code {result.returncode}")
            
            print(result.stdout)
            
            # Determine output file path
            # NetCleave writes to <input_base>_NetCleave.csv in the SAME DIRECTORY as input
            input_base = input_csv_path.rsplit('.', 1)[0]
            generated_output_path = f"{input_base}_NetCleave.csv"
            
            print(f"Looking for generated NetCleave output at: {generated_output_path}")
            
            if os.path.exists(generated_output_path):
                # Read results
                results_df = pd.read_csv(generated_output_path)
                
                # Check for score column
                score_col = 'Cleavage site prediction score'
                if score_col not in results_df.columns:
                     # Attempt to find it
                     for col in results_df.columns:
                         if 'score' in col.lower():
                             score_col = col
                             break
                
                if score_col in results_df.columns:
                    scores = results_df[score_col].values
                    
                    # Initialize column
                    data['cleavage_score'] = 0.0 # Default/Low prob
                    
                    if len(scores) == len(valid_indices):
                        data.loc[valid_indices, 'cleavage_score'] = scores
                    else:
                        print(f"Warning: Result count {len(scores)} mismatch with input {len(valid_indices)}")
                else:
                    print(f"Warning: Score column not found in results: {results_df.columns}")
                
                # Clean up local output file
                if os.path.exists(generated_output_path):
                    os.remove(generated_output_path)
            else:
                print("Error: Output file not found.")
                raise FileNotFoundError(f"NetCleave output file {generated_output_path} not found.")

        finally:
            if os.path.exists(input_csv_path):
                os.remove(input_csv_path)

        self.save_results(data, "cleavage_likelihood.csv", **kwargs)

        return data
