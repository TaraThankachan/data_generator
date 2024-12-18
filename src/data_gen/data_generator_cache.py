import time
import pandas as pd
import numpy as np
import logging
from collections import OrderedDict
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, max_size: int, ttl: int):
        """
        Initialize the cache.

        Parameters:
        - max_size: Maximum number of items allowed in the cache.
        - ttl: Time-to-live for each cache entry in seconds.
        """
        self.cache = OrderedDict()  # OrderedDict to maintain insertion order for eviction.
        self.max_size = max_size    # Maximum size of the cache.
        self.ttl = ttl              # Time-to-live for each cache entry in seconds.

    def get(self, key: Tuple) -> pd.DataFrame:
        """
        Retrieve an item from the cache if it exists and is not expired.

        Parameters:
        - key: The key to look up in the cache.

        Returns:
        - The cached value if valid, otherwise None.
        """
        try:
            if key in self.cache:
                value, timestamp = self.cache[key]  # Retrieve the cached value and its timestamp.
                if time.time() - timestamp < self.ttl:  # Check if the cache entry is still valid.
                    logger.info(f"Cache hit for key: {key}")
                    return value
                else:
                    logger.info(f"Cache expired for key: {key}")
            return None  # Return None if the key is not found or the entry is expired.
        except Exception as e:
            logger.error(f"Error retrieving key {key} from cache: {e}")
            return None

    def put(self, key: Tuple, value: pd.DataFrame):
        """
        Store an item in the cache.

        Parameters:
        - key: The unique key for the cache entry.
        - value: The value to store in the cache (e.g., a Pandas DataFrame).
        """
        try:
            self.cache[key] = (value, time.time())  # Add the new item to the cache with the current timestamp.
            if len(self.cache) > self.max_size:  # Check if the cache exceeds its limit.
                evicted_key, _ = self.cache.popitem(last=False)  # FIFO eviction: Remove the oldest entry.
                logger.info(f"Cache size exceeded. Evicted key: {evicted_key}")
        except Exception as e:
            logger.error(f"Error storing key {key} in cache: {e}")

    def clear(self):
        """
        Clear all entries in the cache.
        """
        try:
            self.cache.clear()
            logger.info("Cache cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


def generate_data(mean: float, std: float, rows: int, columns: int) -> pd.DataFrame:
    """
    Simulates generating a DataFrame with random data.

    Parameters:
    - mean: Mean of the random data.
    - std: Standard deviation of the random data.
    - rows: Number of rows in the DataFrame.
    - columns: Number of columns in the DataFrame.

    Returns:
    - A Pandas DataFrame filled with random data.
    """
    try:
        # Simulate a slow process by introducing a delay.
        time.sleep(5)  # This delay represents time-consuming computations or I/O operations.
        # Generate random data using NumPy and return as a Pandas DataFrame.
        data = pd.DataFrame(np.random.normal(mean, std, size=(rows, columns)))
        logger.info(f"Data generated successfully for mean={mean}, std={std}, rows={rows}, columns={columns}")
        return data
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise


class DataGenerator:
    def __init__(self, max_cache_size: int = 10, ttl: int = 300):
        """
        Initialize the DataGenerator with caching capabilities.

        Parameters:
        - max_cache_size: Maximum number of items in the cache.
        - ttl: Time-to-live for each cache entry.
        """
        self.cache = Cache(max_cache_size, ttl)  # Instantiate the Cache class.
        self.last_update_time = 0  # Tracks the last time the cache was invalidated.

    def generate_pd_dataframe(self, mean: float, std: float, rows: int, columns: int) -> pd.DataFrame:
        """
        Generate a Pandas DataFrame, either from the cache or by computing it.

        Parameters:
        - mean: Mean of the random data.
        - std: Standard deviation of the random data.
        - rows: Number of rows in the DataFrame.
        - columns: Number of columns in the DataFrame.

        Returns:
        - A Pandas DataFrame with the generated random data.
        """
        key = (mean, std, rows, columns)  # Create a unique key based on the input parameters.

        try:
            # Try to retrieve the data from the cache.
            data = self.cache.get(key)

            # Time-based invalidation: Clear the cache every hour.
            if time.time() - self.last_update_time > 3600:  # Check if an hour has passed.
                self.cache.clear()
                self.last_update_time = time.time()  # Update the last cache invalidation time.

            if data is None:  # If data is not in the cache or has expired.
                logger.info(f"Cache miss for key: {key}. Generating new data...")
                # Generate the data and store it in the cache.
                data = generate_data(mean, std, rows, columns)
                self.cache.put(key, data)  # Add the newly generated data to the cache.
            else:
                logger.info(f"Returning cached data for key: {key}")
            
            return data  # Return the DataFrame.

        except Exception as e:
            logger.error(f"Error generating DataFrame for key {key}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    generator = DataGenerator()  # Create an instance of the DataGenerator.

    try:
        # Generate a DataFrame (cache miss on the first call).
        df1 = generator.generate_pd_dataframe(10, 2, 100, 5)
        print("First DataFrame:\n", df1)

        # Generate the same DataFrame (cache hit on the second call).
        df2 = generator.generate_pd_dataframe(10, 2, 100, 5)
        print("Second DataFrame:\n", df2)

        # Generate another DataFrame with different parameters (cache miss).
        df3 = generator.generate_pd_dataframe(20, 5, 50, 3)
        print("Third DataFrame:\n", df3)

    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}")
