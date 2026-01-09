from abc import ABC, abstractmethod
import pandas as pd

class PipelineStep(ABC):
    """
    Abstract base class for all genomic pipeline steps.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process the input data and return the transformed data.
        
        Args:
            data (pd.DataFrame): The input data (usually a dataframe of peptides).
            **kwargs: Additional arguments if needed.
            
        Returns:
            pd.DataFrame: The processed dataframe.
        """
        pass

    def save_results(self, data: pd.DataFrame, filename: str, **kwargs):
        """
        Helper to save intermediate results if output_dir is provided.
        """
        import os
        import logging
        
        output_dir = kwargs.get('output_dir')
        if output_dir and data is not None:
             try:
                 if not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                 path = os.path.join(output_dir, filename)
                 data.to_csv(path, index=False)
                 logging.info(f"Step '{self.name}' results saved to {path}")
             except Exception as e:
                 logging.warning(f"Failed to save results for step '{self.name}': {e}")
