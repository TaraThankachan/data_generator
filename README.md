# Extensible Multi-Layer Cache System

## Overview

This project provides an extensible caching system that integrates multiple caching mechanisms, including:
- **In-Memory Cache**: Fast, volatile cache for quick access.
- **Disk Cache**: Persistent, file-based cache.
- **Redis Cache**: Distributed, scalable caching mechanism. Not yet implimented

The system is designed to optimize data retrieval and storage, especially for expensive operations like data generation. It adheres to **SOLID principles** to ensure flexibility, maintainability, and scalability.

---

## Features

1. **Unified Cache Management**: Combines memory, disk, and Redis caching layers into a single, user-friendly interface.
2. **Layered Caching**:
   - Checks memory first (fastest).
   - Falls back to Redis (distributed).
   - Finally checks disk (persistent storage).
3. **Extensibility**: Easily add new caching layers by implementing the `CacheBase` interface.
4. **Efficient Data Retrieval**:
   - Promotes frequently accessed data to faster layers (e.g., memory).
5. **TTL Support**: Automatically expires stale cache entries.

---

## Prerequisites

- Python 3.7+


---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
