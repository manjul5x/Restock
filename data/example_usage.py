"""
Example usage of the standardized data loader.
"""

from loader import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Create loader (singleton)
    loader = DataLoader()
    
    try:
        # Load product master first
        product_master = loader.load_product_master(
            columns=['product_id', 'location_id', 'risk_period']
        )
        print(f"Loaded {len(product_master)} product master records")
        
        # Load filtered outflow data
        outflow = loader.load_outflow(
            product_master=product_master,
            columns=['product_id', 'location_id', 'date', 'demand']
        )
        print(f"Loaded {len(outflow)} outflow records")
        
        # Process data...
        
        # Check cache stats
        stats = loader.get_cache_stats()
        print("\nCache Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clear cache at workflow boundary
        loader.clear_cache()

if __name__ == "__main__":
    main()