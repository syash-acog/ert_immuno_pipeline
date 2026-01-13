#!/usr/bin/env python3
"""
Immunogenicity Risk Pipeline

A computational pipeline for predicting immunogenicity risk of therapeutic proteins.
Analyzes MHC-II binding, cleavage likelihood, structural accessibility, and 
T-cell receptor facing residues to generate comprehensive risk assessments.

Usage:
    python src/main.py --fasta data/input/protein.fasta --pdb data/input/structure.pdb --alleles "HLA-DRB1*03:01,HLA-DRB1*01:01"
"""

import sys
import os
import argparse
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Immunogenicity Risk Pipeline - Predict immunogenic hotspots in therapeutic proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python src/main.py --fasta data/input/protein.fasta
    
    # Run with specific alleles
    python src/main.py --fasta data/input/protein.fasta --alleles "HLA-DRB1*03:01,HLA-DRB1*01:01"
    
    # Run with PDB structure for accessibility analysis
    python src/main.py --fasta data/input/protein.fasta --pdb data/input/structure.pdb

Note:
    - FASTA file is required
    - PDB file is optional (if not provided, accessibility will use default values)
    - Alleles must be HLA-DRB1 format (e.g., HLA-DRB1*03:01)
    - Multiple alleles should be comma-separated
        """
    )
    
    parser.add_argument(
        "--fasta", "-f",
        type=str,
        required=True,
        help="Path to input FASTA file containing protein sequence"
    )
    
    parser.add_argument(
        "--pdb", "-p",
        type=str,
        default=None,
        help="Path to PDB structure file (optional, for structural accessibility)"
    )
    
    parser.add_argument(
        "--alleles", "-a",
        type=str,
        default="HLA-DRB1*03:01,HLA-DRB1*01:01",
        help="Comma-separated list of HLA alleles (default: HLA-DRB1*03:01,HLA-DRB1*01:01)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: data/output)"
    )
    
    parser.add_argument(
        "--peptide-length", "-l",
        type=int,
        default=15,
        help="Peptide length for enumeration (default: 15)"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input files and parameters."""
    # Check FASTA file exists
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        sys.exit(1)
    
    # Check PDB file if provided
    if args.pdb and not os.path.exists(args.pdb):
        print(f"Warning: PDB file not found: {args.pdb}")
        print("Continuing without structural accessibility analysis...")
        args.pdb = None
    
    # Parse alleles
    alleles = [a.strip() for a in args.alleles.split(",")]
    
    # Validate allele format
    valid_alleles = []
    for allele in alleles:
        if allele.startswith("HLA-DRB1*"):
            valid_alleles.append(allele)
        elif allele.startswith("HLA-DQ") or allele.startswith("HLA-DP"):
            print(f"Warning: Allele {allele} is not supported by TEPITOPE (only HLA-DRB1). Skipping.")
        else:
            print(f"Warning: Invalid allele format: {allele}. Expected HLA-DRB1*XX:XX. Skipping.")
    
    if not valid_alleles:
        print("Error: No valid alleles provided. Please use HLA-DRB1 format (e.g., HLA-DRB1*03:01)")
        sys.exit(1)
    
    return valid_alleles


def main():
    """Main pipeline execution."""
    # Parse arguments
    args = parse_args()
    
    # Validate inputs
    alleles = validate_inputs(args)
    
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_fasta = os.path.abspath(args.fasta)
    input_pdb = os.path.abspath(args.pdb) if args.pdb else None
    output_dir = args.output if args.output else os.path.join(base_dir, "data", "output")
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print configuration
    print("=" * 60)
    print("IMMUNOGENICITY RISK PIPELINE")
    print("=" * 60)
    print(f"Input FASTA:     {input_fasta}")
    print(f"Input PDB:       {input_pdb if input_pdb else 'Not provided'}")
    print(f"Alleles:         {', '.join(alleles)}")
    print(f"Peptide Length:  {args.peptide_length}")
    print(f"Output Dir:      {output_dir}")
    print("=" * 60)
    
    # Initialize Steps
    steps = [
        PeptideEnumeration(length=args.peptide_length),
        CleavageLikelihood(),
        StructuralAccessibility(),
        MHCBinding(alleles=alleles),
        RegisterResolution(),
        ImmunogenicityScoring(allele=alleles[0]),
        ResidueRiskMapping(),
        PopulationWeighting()
    ]
    
    # Initialize Pipeline
    pipeline = GenomicPipeline(steps)
    
    # Run pipeline
    start_time = time.time()
    result_df = pipeline.run(
        initial_data=None, 
        fasta_path=input_fasta, 
        pdb_path=input_pdb, 
        output_dir=output_dir
    )
    end_time = time.time()
    
    # Print results
    print("=" * 60)
    if result_df is not None:
        duration = end_time - start_time
        print(f"Pipeline completed successfully in {duration:.2f} seconds.")
        print("")
        print("Final Protein Risk Score:")
        print(result_df.to_string(index=False))
        print("")
        print(f"Output files saved to: {output_dir}")
    else:
        print("Pipeline failed or produced no output.")
    print("=" * 60)


if __name__ == "__main__":
    main()
