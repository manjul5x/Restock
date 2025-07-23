"""
Parallel processing utilities for the forecasting pipeline.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Tuple
import pandas as pd
import logging
from functools import partial
import time
import traceback

from .config import RunnerConfig, BatchConfig


class ParallelProcessor:
    """Handles parallel processing of forecasting batches."""
    
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_workers = min(config.max_workers, mp.cpu_count())
    
    def process_batches_parallel(
        self,
        batches: List[BatchConfig],
        process_function: Callable,
        *args,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process batches in parallel using ProcessPoolExecutor.
        
        Args:
            batches: List of batch configurations
            process_function: Function to process each batch
            *args: Additional arguments for process_function
            **kwargs: Additional keyword arguments for process_function
        
        Returns:
            List of results from each batch
        """
        self.logger.info(f"Starting parallel processing of {len(batches)} batches with {self.max_workers} workers")
        
        results = []
        start_time = time.time()
        
        # Create a partial function with the additional arguments
        process_func = partial(process_function, *args, **kwargs)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_func, batch): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append({
                        'batch_id': batch.batch_id,
                        'status': 'success',
                        'result': result,
                        'product_locations': batch.product_locations
                    })
                    self.logger.info(f"Batch {batch.batch_id} completed successfully")
                except Exception as e:
                    error_msg = f"Batch {batch.batch_id} failed: {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    results.append({
                        'batch_id': batch.batch_id,
                        'status': 'error',
                        'error': str(e),
                        'product_locations': batch.product_locations
                    })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log summary
        successful_batches = sum(1 for r in results if r['status'] == 'success')
        failed_batches = len(results) - successful_batches
        
        self.logger.info(f"Parallel processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Successful batches: {successful_batches}/{len(batches)}")
        self.logger.info(f"Failed batches: {failed_batches}/{len(batches)}")
        
        return results
    
    def process_batches_sequential(
        self,
        batches: List[BatchConfig],
        process_function: Callable,
        *args,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process batches sequentially (useful for debugging).
        
        Args:
            batches: List of batch configurations
            process_function: Function to process each batch
            *args: Additional arguments for process_function
            **kwargs: Additional keyword arguments for process_function
        
        Returns:
            List of results from each batch
        """
        self.logger.info(f"Starting sequential processing of {len(batches)} batches")
        
        results = []
        start_time = time.time()
        
        for batch in batches:
            try:
                self.logger.info(f"Processing batch {batch.batch_id}")
                result = process_function(batch, *args, **kwargs)
                results.append({
                    'batch_id': batch.batch_id,
                    'status': 'success',
                    'result': result,
                    'product_locations': batch.product_locations
                })
                self.logger.info(f"Batch {batch.batch_id} completed successfully")
            except Exception as e:
                error_msg = f"Batch {batch.batch_id} failed: {str(e)}"
                self.logger.error(error_msg)
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                results.append({
                    'batch_id': batch.batch_id,
                    'status': 'error',
                    'error': str(e),
                    'product_locations': batch.product_locations
                })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.logger.info(f"Sequential processing completed in {processing_time:.2f} seconds")
        
        return results
    
    def create_batches(
        self,
        product_locations: List[Tuple[str, str]]
    ) -> List[BatchConfig]:
        """
        Create batches from product-location combinations.
        
        Args:
            product_locations: List of (product_id, location_id) tuples
        
        Returns:
            List of BatchConfig objects
        """
        batches = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(product_locations), batch_size):
            batch_product_locations = product_locations[i:i + batch_size]
            batch_id = i // batch_size + 1
            
            batch_config = BatchConfig(
                batch_id=batch_id,
                product_locations=batch_product_locations,
                config=self.config
            )
            batches.append(batch_config)
        
        self.logger.info(f"Created {len(batches)} batches from {len(product_locations)} product-location combinations")
        
        return batches
    
    def get_processing_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get processing statistics from batch results.
        
        Args:
            results: List of batch processing results
        
        Returns:
            Dictionary with processing statistics
        """
        total_batches = len(results)
        successful_batches = sum(1 for r in results if r['status'] == 'success')
        failed_batches = total_batches - successful_batches
        
        # Calculate average processing time if available
        processing_times = []
        for result in results:
            if 'result' in result and isinstance(result['result'], dict):
                if 'processing_time' in result['result']:
                    processing_times.append(result['result']['processing_time'])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_batches': total_batches,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'success_rate': successful_batches / total_batches if total_batches > 0 else 0,
            'average_processing_time': avg_processing_time,
            'total_product_locations': sum(len(r['product_locations']) for r in results)
        }


def process_batch_with_logging(
    batch: BatchConfig,
    process_function: Callable,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a single batch with logging and error handling.
    
    Args:
        batch: Batch configuration
        process_function: Function to process the batch
        *args: Additional arguments for process_function
        **kwargs: Additional keyword arguments for process_function
    
    Returns:
        Dictionary with processing results
    """
    import logging
    
    # Set up logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(f"Batch_{batch.batch_id}")
    
    start_time = time.time()
    
    try:
        logger.info(f"Starting batch {batch.batch_id} with {len(batch.product_locations)} product-location combinations")
        
        # Process the batch
        result = process_function(batch, *args, **kwargs)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Batch {batch.batch_id} completed successfully in {processing_time:.2f} seconds")
        
        return {
            'processing_time': processing_time,
            'data': result
        }
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.error(f"Batch {batch.batch_id} failed after {processing_time:.2f} seconds: {str(e)}")
        raise
