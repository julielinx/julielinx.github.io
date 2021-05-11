---
title: "Entry G5.5: Analysis Metrics"
categories:
  - Blog
tags:
  - graph
  - neo4j
---

I am 100% adding this entry retroactively. And yes, I dated it wrong so it would show up in the right order in the Post list. That's the nice thing about digital diaries: you can insert things in after the fact. It wasn't until I was working on entries G12 and G13 that I realized I hadn't posted all the analysis metrics I plan to address.

There is no guarantee that I'll address all metrics under a single category in one post and no guarantee I'll post them in the order listed. Although, I may update the list after the entries are live to reflect how they end up grouped.

I put together this list from my research into network theory, including:

- [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098) by Mark Newman
- The online book [Network Science](http://networksciencebook.com/) by Albert-László Barabási as well as other books by him and his [online material](https://barabasi.com/publications)
- [Graph Algorithms](https://neo4j.com/graph-algorithms-book/) by Mark Needham and Amy Hodler
- [Graph Databases](https://neo4j.com/graph-databases-book/) by Ian Robinson, Jim Webber and Emil Eifrem
- [Connected](https://neo4j.com/books/connected/) by Nicholas Christakis and James Fowler
- [Graph People Blog](https://tbgraph.wordpress.com/)
- Courses/videos/tutorials/blog posts through [Neo4j](https://neo4j.com/graphacademy/)
- Others that I've obviously forgotten about by name

I'm not going to go into any detail on any of these topics in this entry. I'll cover what each metric is and how it applies to data in the relevant entries.

If I'm feeling ambitious, I might come back and put the links to the relevant entries once they've been posted, like I've done for Global Metrics.

### Global Metrics

1. [Counts: Entry G6](https://julielinx.github.io/blog/g06_global_counts/)
  - Node count
  - Isolates count and percent
  - Relationship count
2. [Density and Diameter: Entry G7](https://julielinx.github.io/blog/g07_global_density_diameter/)
  - Number of possible relationships
  - Global density
  - Diameter
3. [Components: Entry G8](https://julielinx.github.io/blog/g08_components/)
  - Component count
  - Component size
  - Component percent

### Local Metrics

1. Degree Descriptive Statistics and Distribution Charts
  - [Entry 10: Local Metrics](https://julielinx.github.io/blog/g10_local_metrics/): Unweighted and weighted degrees first pass
  - [Entry 12: Degree Comparison](https://julielinx.github.io/blog/g12_degree_comparison/): Unweighted degrees refined
  - [Entry 13: Weighted Degree Comparison](https://julielinx.github.io/blog/g13_weighted_degree_comparison/): Weighted degrees refined
2. Weighted degree count
3. Weighted degree descriptive statistics
4. Weighted degree distribution

### Density and Nearest Neighbors

1. Local density at various step levels (nearest neighbors, next nearest neighbors)
2. Number of nearest claim neighbors
3. Number of next nearest claim neighbors
4. Number steps to nearest fraud
5. Count of fraud at various step levels
6. Percent of fraud at various step levels
7. Distribution of fraud at various step levels
8. Descriptive statistics for fraud at various step levels

### Shortest Path

1. Shortest path
2. Shortest path descriptive statistics
3. Shortest path distribution
4. Shortest path to fraud
5. Distribution of shortest path to fraud
6. Descriptive statistics for shortest path to fraud

### Triangles

1. Triangles / triads
2. Distribution of triangles
3. Triangles descriptive statistics

### Tours / Cycles

1. Tours / cycles
2. Distribution of tours
3. Tours descriptive statistics

### Clustering Coefficient

1. Clustering coefficient
2. Global clustering coefficient
3. Local clustering coefficient
4. Distribution of clustering coefficient
5. Clustering coefficient descriptive statistics

### Centralities

1. Degree Centrality
2. Betweenness Centrality
3. Closeness Centrality
4. PageRank Centrality
5. Eigenvector centrality

### K-cores

Not sure what I want to do with K-cores. I'll explore it more when I get to it.

### Other

1. Community Detection
2. Reciprocity
3. Network link analysis

### Up Next

[Global Graph Counts](https://julielinx.github.io/blog/g06_global_counts/)