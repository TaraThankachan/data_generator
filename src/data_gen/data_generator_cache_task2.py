import time
import pandas as pd
import numpy as np
import logging
import os
import pickle
from collections import OrderedDict
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cache:
    def __init__(self, max_size: int, ttl: int, cache_dir: str = "./cache"):
        """
        Initialize the cache with both memory and disk-based storage.

        Parameters:
        - max_size: Maximum number of items allowed in the cache.
        - ttl: Time-to-live for each cache entry in seconds.
        - cache_dir: Directory path to store cached files.
        """
        self.cache = OrderedDict()  # In-memory cache.
        self.max_size = max_size  # Maximum cache size.
        self.ttl = ttl  # Time-to-live for cache entries.
        self.cache_dir = cache_dir  # Directory for disk-based cache.

        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_disk_cache_file(self, key: Tuple) -> str:
        """
        Generate a unique file path for a cache key.
        The file is used to store the serialized version of the cache data on the disk.

        Parameters:
        - key: The unique key for the cache entry.

        Returns:
        - File path as a string.
        """
        return os.path.join(self.cache_dir, f"{hash(key)}.pkl") 

    def get(self, key: Tuple) -> pd.DataFrame:
        """
        Retrieve an item from the in- memory cache if it exists and is not expired.

        Parameters:
        - key: The key to look up in the cache.

        Returns:
        - The cached value if valid, otherwise None.
        """
        try:
            # Check in-memory cache
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.info(f"Cache hit (memory) for key: {key}")
                    return value
                else:
                    logger.info(f"Cache expired (memory) for key: {key}")
                    del self.cache[key]

            # Check disk cache
            # Check if the corresponding cache file exists on the disk.
            cache_file = self._get_disk_cache_file(key)
            if os.path.exists(cache_file): 
                with open(cache_file, 'rb') as f:
                    data, timestamp = pickle.load(f)
                if time.time() - timestamp < self.ttl:
                    logger.info(f"Cache hit (disk) for key: {key}")
                    return data
                else:
                    logger.info(f"Cache expired (disk) for key: {key}")
                    os.remove(cache_file)

            return None
        except Exception as e:
            logger.error(f"Error retrieving key {key} from cache: {e}")
            return None

    def put(self, key: Tuple, value: pd.DataFrame):
        """
        Store an item in the cache (both in memory and disk).

        Parameters:
        - key: The unique key for the cache entry.
        - value: The value to store in the cache (e.g., a Pandas DataFrame).
        """
        try:
            # Add to memory cache
            self.cache[key] = (value, time.time()) # Add the new item to the cache with the current timestamp.
            if len(self.cache) > self.max_size: # Check if the cache exceeds its limit.
                evicted_key, _ = self.cache.popitem(last=False) # FIFO eviction
                logger.info(f"Cache size exceeded. Evicted key: {evicted_key}")

            # Save to disk cache
            cache_file = self._get_disk_cache_file(key)# Generate a unique file path for the cache key using the hash function.
            with open(cache_file, 'wb') as f: # Open the file in binary write mode ('wb') to store the cache data.
                pickle.dump((value, time.time()), f)  # Serialize the value and the current timestamp using pickle, and write them to the file.
        except Exception as e:
            logger.error(f"Error storing key {key} in cache: {e}")

    def clear(self):
        """
        Clear all entries in the cache (memory and disk).
        """
        try:
            # Clear memory cache
            self.cache.clear()
            # Clear disk cache
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if file.endswith(".pkl") and os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cache cleared successfully (memory and disk).")
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
        time.sleep(5)
        data = pd.DataFrame(np.random.normal(mean, std, size=(rows, columns)))
        logger.info(f"Data generated for mean={mean}, std={std}, rows={rows}, columns={columns}")
        return data
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise


class DataGenerator:
    def __init__(self, max_cache_size: int = 10, ttl: int = 300, cache_dir: str = "./cache"):
        """
        Initialize the DataGenerator with caching capabilities.

        Parameters:
        - max_cache_size: Maximum number of items in the cache.
        - ttl: Time-to-live for each cache entry.
        - cache_dir: Directory path to store disk-based cache files.
        """
        self.cache = Cache(max_cache_size, ttl, cache_dir)
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
        key = (mean, std, rows, columns)
        try:
            # Time-based cache invalidation
            if time.time() - self.last_update_time > 3600:
                self.cache.clear()
                self.last_update_time = time.time()

            # Retrieve from cache or generate new data
            data = self.cache.get(key)
            if data is None:
                logger.info(f"Cache miss for key: {key}. Generating new data...")
                data = generate_data(mean, std, rows, columns)
                self.cache.put(key, data)
            else:
                logger.info(f"Returning cached data for key: {key}")
            return data
        except Exception as e:
            logger.error(f"Error generating DataFrame: {e}")
            raise


# Example usage
if __name__ == "__main__":
    generator = DataGenerator()

    # First run: cache miss
    df1 = generator.generate_pd_dataframe(10, 2, 100, 5)
    print(df1)

    # Second run: cache hit
    df2 = generator.generate_pd_dataframe(10, 2, 100, 5)
    print(df2)

    # Third run: different parameters (cache miss)
    df3 = generator.generate_pd_dataframe(20, 5, 50, 3)
    print(df3)
