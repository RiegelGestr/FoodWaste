# get_distance_stores.py

This script computes travel distances and times between stores using **OSRM** (Open Source Routing Machine), a high-performance routing engine based on OpenStreetMap data.

## Overview

The script queries a local OSRM backend to calculate routing distances and travel times between store locations. It supports batch distance matrix computations and is optimized for handling large numbers of store pairs.

## Prerequisites

### OSRM Backend Setup

This script requires a running OSRM routing engine. The easiest way to set this up is using **Docker**.

#### 1. Download OpenStreetMap Data

Get the OSM data extract for your region from [Geofabrik](https://download.geofabrik.de). For example, to download Denmark:

```bash
wget https://download.geofabrik.de/europe/denmark-latest.osm.pbf
```

Available regions: [download.geofabrik.de](https://download.geofabrik.de) - organized by continent and country.

#### 2. Preprocess the OSM Data with Docker

Extract, partition, and customize the routing data. Run these commands in sequence in the directory containing `denmark-latest.osm.pbf`:

```bash
# Extract the routing graph
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract denmark-latest.osm.pbf || echo "osrm-extract failed"

# Partition the data for multi-level Dijkstra (MLD) algorithm
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition denmark-latest.osrm || echo "osrm-partition failed"

# Customize the data for routing optimization
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize denmark-latest.osrm || echo "osrm-customize failed"
```

#### 3. Start the OSRM Routing Server

Start the OSRM backend server on port 5001 (adjust as needed):

```bash
docker run -t -i -p 5001:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --max-matching-size=1000000 --max-alternatives=10 --algorithm MLD denmark-latest.osrm
```

**Key parameters:**
- `-p 5001:5000`: Maps Docker container port 5000 to localhost port 5001
- `--max-matching-size=1000000`: Maximum coordinates for map matching
- `--max-alternatives=10`: Maximum alternative routes returned
- `--algorithm MLD`: Uses multi-level Dijkstra for fast routing

The server will be accessible at `http://localhost:5001` and ready to receive routing requests.

## Usage

```python
from get_distance_stores import compute_distances

# Example: compute distance from coordinate A to coordinate B
distances = compute_distances(
    origins=[(lon1, lat1), (lon2, lat2)],
    destinations=[(lon3, lat3), (lon4, lat4)],
    profile='car'  # 'car', 'bike', 'foot'
)
```

## Configuration

Update the OSRM backend URL in the script if using a different port or host:

```python
OSRM_URL = "http://localhost:5001"
```

## Supported Profiles

- **car**: Car routing (default)
- **bike**: Bicycle routing
- **foot**: Pedestrian routing

## Dependencies

See `requirements.txt` for Python dependencies. The script typically requires:
- `requests`: for HTTP calls to OSRM
- `pandas`: for data manipulation
- `numpy`: for numerical operations

## Further Information

- **OSRM GitHub**: [github.com/Project-OSRM/osrm-backend](https://github.com/Project-OSRM/osrm-backend)
- **OSRM API Documentation**: [project-osrm.org/docs](https://project-osrm.org/docs/v5.5.1/api/)
- **Geofabrik Data**: [download.geofabrik.de](https://download.geofabrik.de)
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com)

## Troubleshooting

**Server not responding:**
- Ensure Docker container is running: `docker ps`
- Verify port mapping: `docker port <container_id>`
- Test connectivity: `curl http://localhost:5001/status`

**Data preprocessing failed:**
- Check available disk space for `.osm.pbf` processing
- Ensure Docker has sufficient memory allocation
- Verify file paths are correct

**Slow routing requests:**
- Ensure MLD algorithm is being used (`--algorithm MLD` flag)
- Consider caching distance matrix results
- Reduce request batch size if experiencing timeouts