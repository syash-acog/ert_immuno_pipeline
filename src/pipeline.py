import pandas as pd
from typing import List
from .steps.base import PipelineStep

class GenomicPipeline:
    """
    Orchestrator for the genomic pipeline.
    """

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, initial_data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Run all steps in the pipeline sequentially.
        
        Args:
            initial_data (pd.DataFrame): Initial data to start the pipeline with (optional).
            **kwargs: Shared context or arguments passed to all steps.
            
        Returns:
            pd.DataFrame: The final result after all steps.
        """
        data = initial_data
        
        for step in self.steps:
            print(f"Running step: {step.name}")
            data = step.process(data, **kwargs)
            print(f"Step {step.name} completed. Data shape: {data.shape if data is not None else 'None'}")

        return data
