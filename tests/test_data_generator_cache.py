import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import time
from src.data_gen.data_generator_cache import DataGenerator, Cache, generate_data

class TestCache(unittest.TestCase):
    def setUp(self):
        # Initialize a small cache for testing purposes.
        self.cache = Cache(max_size=3, ttl=2)

    def test_cache_put_and_get(self):
        """
        Test that items can be added to and retrieved from the cache correctly.
        """
        key = (1, 2, 3)
        value = pd.DataFrame(np.random.rand(3, 3))
        self.cache.put(key, value)
        cached_value = self.cache.get(key)
        self.assertTrue(cached_value.equals(value), "Cached value does not match the expected value.")

    def test_cache_eviction(self):
        """
        Test that the oldest item is evicted when the cache exceeds its maximum size.
        """
        for i in range(4):
            key = (i,)
            value = pd.DataFrame(np.random.rand(3, 3))
            self.cache.put(key, value)
        self.assertEqual(len(self.cache.cache), 3, "Cache did not evict the oldest entry when exceeding max size.")
        self.assertNotIn((0,), self.cache.cache, "Oldest cache entry was not evicted.")

    def test_cache_expiration(self):
        """
        Test that cache entries are removed after their TTL has expired.
        """
        key = (1, 2, 3)
        value = pd.DataFrame(np.random.rand(3, 3))
        self.cache.put(key, value)
        time.sleep(3)  # Wait for the entry to expire.
        cached_value = self.cache.get(key)
        self.assertIsNone(cached_value, "Expired cache entry was not removed.")

    def test_clear_cache(self):
        """
        Test that the cache can be cleared successfully.
        """
        key = (1, 2, 3)
        value = pd.DataFrame(np.random.rand(3, 3))
        self.cache.put(key, value)
        self.cache.clear()
        self.assertEqual(len(self.cache.cache), 0, "Cache was not cleared successfully.")

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        # Initialize the DataGenerator with a cache size of 5 and TTL of 5 seconds.
        self.generator = DataGenerator(max_cache_size=5, ttl=5)

    @patch('src.data_gen.data_generator_cache.generate_data')
    def test_generate_pd_dataframe_cache_hit(self, mock_generate_data):
        """
        Test that cached data is returned on a cache hit without regenerating the data.
        """
        mock_data = pd.DataFrame(np.random.rand(5, 5))
        mock_generate_data.return_value = mock_data

        # First call - cache miss.
        key = (10, 2, 100, 5)
        df1 = self.generator.generate_pd_dataframe(*key)
        mock_generate_data.assert_called_once()

        # Second call - cache hit.
        df2 = self.generator.generate_pd_dataframe(*key)
        self.assertTrue(df1.equals(df2), "DataFrames do not match for cache hit.")
        self.assertEqual(mock_generate_data.call_count, 1, "Data was regenerated instead of using the cache.")

    @patch('src.data_gen.data_generator_cache.generate_data')
    def test_generate_pd_dataframe_cache_miss(self, mock_generate_data):
        """
        Test that data is generated correctly when not present in the cache.
        """
        mock_data = pd.DataFrame(np.random.rand(5, 5))
        mock_generate_data.return_value = mock_data

        # Generate data with one key.
        df1 = self.generator.generate_pd_dataframe(10, 2, 100, 5)
        mock_generate_data.assert_called_once()

        # Generate data with a different key.
        df2 = self.generator.generate_pd_dataframe(20, 5, 50, 3)
        self.assertNotEqual(df1.equals(df2), "Generated data should be different for different keys.")
        self.assertEqual(mock_generate_data.call_count, 2, "Data was not generated for the second key.")

    @patch('src.data_gen.data_generator_cache.generate_data')
    def test_cache_invalidation_after_ttl(self, mock_generate_data):
        """
        Test that expired cache entries are invalidated and regenerated.
        """
        mock_data = pd.DataFrame(np.random.rand(5, 5))
        mock_generate_data.return_value = mock_data

        # Generate data and let it expire.
        key = (10, 2, 100, 5)
        self.generator.generate_pd_dataframe(*key)
        time.sleep(6)  # Wait for the cache entry to expire.

        # Generate data again.
        self.generator.generate_pd_dataframe(*key)
        self.assertEqual(mock_generate_data.call_count, 2, "Cache did not expire after TTL.")

if __name__ == "__main__":
    unittest.main()
