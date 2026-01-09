import pandas as pd
from Bio import SeqIO
from .base import PipelineStep

class PeptideEnumeration(PipelineStep):
    """
    Step 1: Generates k-mers (peptides) from a protein sequence.
    """

    def __init__(self, length: int = 15):
        super().__init__("Peptide Enumeration")
        self.length = length

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Reads a FASTA file from kwargs and generates k-mers.
        Input 'data' is ignored as this is the starting step.
        """
        fasta_path = kwargs.get('fasta_path')
        if not fasta_path:
            raise ValueError("fasta_path is required for PeptideEnumeration")

        try:
            record = SeqIO.read(fasta_path, "fasta")
            sequence = str(record.seq)
        except Exception as e:
            raise ValueError(f"Failed to read FASTA file at {fasta_path}: {e}")

        peptides_with_pos = self.generate_peptides(sequence, self.length)
        
        # Create a DataFrame
        # We include protein_id and protein_seq as required by NetCleave pred_input=3
        df = pd.DataFrame(peptides_with_pos, columns=['peptide_seq', 'start_pos'])
        df['protein_id'] = record.id
        df['protein_seq'] = sequence
        
        self.save_results(df, "peptides_enumeration.csv", **kwargs)
        
        return df

    def generate_peptides(self, sequence: str, k: int) -> list:
        """
        Generate all k-mers from the sequence with their start positions.
        Returns list of tuples (peptide, start_pos).
        """
        if len(sequence) < k:
            return []
        
        return [(sequence[i:i+k], i) for i in range(len(sequence) - k + 1)]
