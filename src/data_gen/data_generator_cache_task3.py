import time
import pandas as pd
import numpy as np
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Abstract Base Class for Cache to enforce implementation standards
class CacheBase(ABC):
    @abstractmethod
    def get(self, key: Tuple):
        """Retrieve an item from the cache."""
        pass

    @abstractmethod
    def put(self, key: Tuple, value: pd.DataFrame):
        """Store an item in the cache."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all entries in the cache."""
        pass


# In-Memory Cache Implementation using OrderedDict
class InMemoryCache(CacheBase):
    def __init__(self, max_size: int, ttl: int):
        """
        Initialize an in-memory cache.

        Parameters:
        - max_size: Maximum number of items in the cache.
        - ttl: Time-to-live (in seconds) for each cache entry.
        """
        self.cache = OrderedDict()  # Maintains insertion order for FIFO eviction
        self.max_size = max_size    # Cache size limit
        self.ttl = ttl              # Time-to-live for cache entries

    def get(self, key: Tuple):
        """
        Retrieve an item from the in-memory cache.

        Parameters:
        - key: Unique identifier for the cache entry.

        Returns:
        - Cached DataFrame if it exists and is not expired; otherwise, None.
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:  # Check if entry has expired
                logger.info(f"Cache hit (memory) for key: {key}")
                return value
            else:
                logger.info(f"Cache expired (memory) for key: {key}")
                del self.cache[key]  # Remove expired entry
        return None

    def put(self, key: Tuple, value: pd.DataFrame):
        """
        Add a new item to the in-memory cache.

        Parameters:
        - key: Unique identifier for the cache entry.
        - value: DataFrame to be cached.
        """
        self.cache[key] = (value, time.time())  # Store value with timestamp
        if len(self.cache) > self.max_size:  # Enforce cache size limit
            evicted_key, _ = self.cache.popitem(last=False)  # Evict oldest item FIFO
            logger.info(f"Cache size exceeded. Evicted key: {evicted_key}")

    def clear(self):
        """Clear all in-memory cache entries."""
        self.cache.clear()
        logger.info("Memory cache cleared.")


# Disk Cache Implementation for Persistent Caching
class DiskCache(CacheBase):
    def __init__(self, cache_dir: str = "./cache", ttl: int = 300):
        """
        Initialize a disk-based cache.

        Parameters:
        - cache_dir: Directory where cached files are stored.
        - ttl: Time-to-live (in seconds) for each cache file.
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        if not os.path.exists(self.cache_dir):  # Create cache directory if not exists
            os.makedirs(self.cache_dir)

    def _get_disk_cache_file(self, key: Tuple) -> str:
        """
        Generate a file path for a given cache key.

        Parameters:
        - key: Unique identifier for the cache entry.

        Returns:
        - File path as a string.
        """
        return os.path.join(self.cache_dir, f"{hash(key)}.pkl")

    def get(self, key: Tuple):
        """
        Retrieve an item from the disk cache.

        Parameters:
        - key: Unique identifier for the cache entry.

        Returns:
        - Cached DataFrame if it exists and is not expired; otherwise, None.
        """
        cache_file = self._get_disk_cache_file(key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data, timestamp = pickle.load(f)
            if time.time() - timestamp < self.ttl:  # Check if cache is still valid
                logger.info(f"Cache hit (disk) for key: {key}")
                return data
            else:
                logger.info(f"Cache expired (disk) for key: {key}")
                os.remove(cache_file)  # Remove expired cache file
        return None

    def put(self, key: Tuple, value: pd.DataFrame):
        """
        Add a new item to the disk cache.

        Parameters:
        - key: Unique identifier for the cache entry.
        - value: DataFrame to be cached.
        """
        cache_file = self._get_disk_cache_file(key)
        with open(cache_file, 'wb') as f:
            pickle.dump((value, time.time()), f)  # Save value with timestamp

    def clear(self):
        """Clear all disk cache entries."""
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            if file.endswith(".pkl") and os.path.isfile(file_path):
                os.remove(file_path)  # Remove cache files
        logger.info("Disk cache cleared.")


# Unified Cache Handler that combines Memory and Disk Cache
class UnifiedCache:
    def __init__(self, memory_cache: InMemoryCache, disk_cache: DiskCache):
        """
        Combine memory and disk caches for efficient caching.

        Parameters:
        - memory_cache: An instance of InMemoryCache.
        - disk_cache: An instance of DiskCache.
        """
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache

    def get(self, key: Tuple):
        """
        Retrieve data from memory first, then disk if not found.

        Parameters:
        - key: Unique identifier for the cache entry.

        Returns:
        - Cached DataFrame if found; otherwise, None.
        """
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        value = self.disk_cache.get(key)
        if value is not None:
            self.memory_cache.put(key, value)  # Promote to memory cache
        return value

    def put(self, key: Tuple, value: pd.DataFrame):
        """Add data to both memory and disk caches."""
        self.memory_cache.put(key, value)
        self.disk_cache.put(key, value)

    def clear(self):
        """Clear both memory and disk caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()


# Function to Simulate Data Generation
def generate_data(mean: float, std: float, rows: int, columns: int) -> pd.DataFrame:
    """
    Generate a DataFrame with random data.

    Parameters:
    - mean: Mean of the data.
    - std: Standard deviation of the data.
    - rows: Number of rows.
    - columns: Number of columns.

    Returns:
    - DataFrame with generated data.
    """
    time.sleep(5)  # Simulate slow operation
    return pd.DataFrame(np.random.normal(mean, std, size=(rows, columns)))


# DataGenerator Class
class DataGenerator:
    def __init__(self, cache: UnifiedCache):
        """Initialize with a unified cache."""
        self.cache = cache
        self.last_update_time = 0

    def generate_pd_dataframe(self, mean: float, std: float, rows: int, columns: int) -> pd.DataFrame:
        """
        Generate or retrieve cached DataFrame.

        Parameters:
        - mean: Mean of the data.
        - std: Standard deviation of the data.
        - rows: Number of rows.
        - columns: Number of columns.

        Returns:
        - Cached or newly generated DataFrame.
        """
        key = (mean, std, rows, columns)
        if time.time() - self.last_update_time > 3600:
            self.cache.clear()  # Clear cache periodically
            self.last_update_time = time.time()

        data = self.cache.get(key)
        if data is None:
            logger.info(f"Cache miss for key: {key}. Generating new data...")
            data = generate_data(mean, std, rows, columns)
            self.cache.put(key, data)
        else:
            logger.info(f"Returning cached data for key: {key}")
        return data


# Example Usage
if __name__ == "__main__":
    memory_cache = InMemoryCache(max_size=10, ttl=300)
    disk_cache = DiskCache(cache_dir="./cache", ttl=300)
    unified_cache = UnifiedCache(memory_cache, disk_cache)

    generator = DataGenerator(cache=unified_cache)

    # First run: cache miss
    df1 = generator.generate_pd_dataframe(10, 2, 100, 5)
    print(df1)

    # Second run: cache hit
    df2 = generator.generate_pd_dataframe(10, 2, 100, 5)
    print(df2)

    # Third run: different parameters (cache miss)
    df3 = generator.generate_pd_dataframe(20, 5, 50, 3)
    print(df3)
