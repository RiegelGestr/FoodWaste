# Mapping Regional Disparities in Discounted Grocery Products

This repository contains data and code to reproduce the analysis presented in the paper **"Mapping Regional Disparities in Discounted Grocery Products"** by Antonio Desiderio, Alessia Galdeman, Franziska Bauerlein, and Sune Lehmann.

## Abstract

Food waste represents a major challenge to global climate resilience, accounting for almost 10% of annual greenhouse gas emissions. The retail sector is a critical player, mediating product flows between producers and consumers, where supply chain inefficiencies can shape which items are put on sale. Yet how these dynamics vary across geographic contexts remains largely unexplored.
Here, we analyze data from Denmark's largest retail group on near-expiry products put on sale. We uncover the geospatial variations using a dual-clustering approach. We identify multi-scale spatial relationships in retail organization by correlating store clusteringâ€”measured using shortest-path distances along the street networkâ€”with product clustering based on promotion co-occurrence patterns. Using a bipartite network approach, we identify three regional store clusters, and use percolation thresholds to corroborate the scale of their spatial separation.
We find that stores in rural communities put meat and dairy products on sale up to 2.2 times more frequently than metropolitan areas. In contrast, we find that metropolitan and capital regions lean toward convenience products, which have more balanced nutritional profiles but less favorable environmental impacts.
By linking geographic context to retail inventory, we provide evidence that reducing food waste requires interventions tailored to local retail dynamics, highlighting the importance of region-specific sustainability strategies.

## Repository Structure

The repository is organized as follows:

### `data/`
Contains both raw data and intermediate processing outputs:
- **Raw data**: Original offers data from the retail group (See below Data Availability Statement)
- **Intermediate data**: Processed outputs including store-to-store distances computed using OSRM (more info inside the folder) and results from the Bipartite Configuration Model

### `src/`
Contains the core computational scripts and modules:
- **OSRM distance computation**: Scripts for calculating shortest-path distances between stores using OpenStreetMap Routing Machine (OSRM)
- **Bipartite Configuration Model**: Implementation of the null model for bipartite network analysis

### `scripts/`
Contains Python scripts to generate the figures and panels presented in the paper:
- Each script corresponds to specific analyses and visualizations
- Scripts produce publication-ready figures

### `figures/`
Contains individual figure panels and complete figures:
- Panels were created and arranged using Gephi for network visualizations
- All figures are included as high-resolution images

## Data Availability Statement

This repository contains a **subsample of the offers per day** from the original retail dataset. The raw data has been filtered and processed to respect data privacy and confidentiality agreements with the retail group. The subsample includes sufficient data to reproduce all analyses and visualizations presented in the paper.

## Requirements

All Python dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Questions and Support

For questions, issues, or requests related to this repository, please:
- Create an issue on the GitHub repository
- Contact the authors by email (in the paper)

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Authors

- **Antonio Desiderio**
- **Alessia Galdeman**
- **Franziska Bauerlein**
- **Sune Lehmann**
