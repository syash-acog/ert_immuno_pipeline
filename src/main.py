import sys
import os
import pandas as pd
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline import GenomicPipeline
from src.steps.peptide_enumeration import PeptideEnumeration
from src.steps.cleavage_likelihood import CleavageLikelihood
from src.steps.structural_accessibility import StructuralAccessibility
from src.steps.mhc_binding import MHCBinding
from src.steps.register_resolution import RegisterResolution
from src.steps.immunogenicity_scoring import ImmunogenicityScoring
from src.steps.residue_risk_mapping import ResidueRiskMapping
from src.steps.population_weighting import PopulationWeighting

def main():
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_fasta = os.path.join(base_dir, "data", "input", "protein.fasta")
    output_dir = os.path.join(base_dir, "data", "output")
    
    # Initialize Steps
    steps = [
        PeptideEnumeration(length=15),
        CleavageLikelihood(),
        StructuralAccessibility(),
        MHCBinding(alleles=["HLA-DRB1*04:01"]),
        RegisterResolution(),
        ImmunogenicityScoring(allele="HLA-DRB1*04:01"),
        ResidueRiskMapping(),
        PopulationWeighting()
    ]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Starting pipeline...")
    print(f"Input: {input_fasta}")
    print(f"Output Directory: {output_dir}")
    
    # Initialize Pipeline
    pipeline = GenomicPipeline(steps)
    
    # pass fasta_path and pdb_path in kwargs
    # Step 1 uses fasta_path
    # Step 3 uses pdb_path (if available)
    input_pdb = os.path.join(base_dir, "data", "input", "2NYN.pdb")
    
    start_time = time.time()
    result_df = pipeline.run(initial_data=None, fasta_path=input_fasta, pdb_path=input_pdb, output_dir=output_dir)
    end_time = time.time()
    
    # Save output
    if result_df is not None:
        duration = end_time - start_time
        print(f"Pipeline completed successfully in {duration:.2f} seconds.")
        print("Final Protein Risk Score:")
        print(result_df)
    else:
        print("Pipeline failed or produced no output.")

if __name__ == "__main__":
    main()
