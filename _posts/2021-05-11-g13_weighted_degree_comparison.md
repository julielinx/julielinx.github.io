---
title: "Entry G13: Weighted Degree Comparison"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

Like [Entry G12](https://julielinx.github.io/blog/g12_degree_comparison/) this is a redo of part of [Entry G10](https://julielinx.github.io/blog/g10_local_metrics/). This entry addresses the weighted degrees. If you need a reminder as to what weighted relationships are see [Entry G4: Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/).

Unlike Entry G12, I only needed one notebook to examine the weighted degrees. It can be found in the [G13 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/13a_nb_weighted_degree_comparison.ipynb).

The changes from the Entry 10 notebooks to this one include:

- Used a multigraph instead of three separate graph instances
- Included `OPTIONAL MATCH` in the summary statistic pattern match queries
- Put results into DataFrames for easier comparison
- Added Comic to Comic summary statistics and distribution charts

## Used a Multigraph

Just like with [Entry G12](https://julielinx.github.io/blog/g12_degree_comparison/), using the multigraph allowed me to query the same information against each of my graph models and easily compare the results. This allowed me to discover that the data in the Mixed Model loaded incorrectly.

There were over 100 relationships that were unaccounted for in the Mixed Model once I summed the weighted relationships. Since the Mixed Model is a stepping stone to the Unimodal Model, I knew that some of the relationships must have errored out. I cleared out the Mixed Modal graph with `MATCH (n) DETACH DELETE n` and reloaded the graph using the code from [Entry G11's notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/11_nb_create_multigraphdb.ipynb). This fixed the discrepancy. It also fixed the discrepancy noted in [Entry G12](https://julielinx.github.io/blog/g12_degree_comparison/) for the number of `KNOWS` relationships in the Unimodal and Mixed Models.

Now, just because I can run a query against all three models doesn't mean I should. If you look in the [G13 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/13a_nb_weighted_degree_comparison.ipynb), you'll notice that I ran Hero to Hero relationships for all three models, but only ran Comic to Comic relationships for the Bimodal and Mixed Models. This is because all Comic information was removed from the Unimodal Model when we projected it.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/projection.png?raw=true'>

## Included OPTIONAL MATCH

Just as explained in [Entry G12](https://julielinx.github.io/blog/g12_degree_comparison/) and earlier in [Entry G10](https://julielinx.github.io/blog/g10_local_metrics/), using `OPTIONAL MATCH` instead of the more restrictive `MATCH` allows the query to find isolate nodes (nodes that have no relationships to other nodes). This gives a more complete picture when examining the summary statistics and distribution charts.

## Put Results in DataFrames

I can't stress enough how helpful it is to have the results in a DataFrame instead of spread out across multiple cells. As an added bonus the formatting is the same every time instead of sometimes putting results on the same line and sometimes putting each result on its own line. Also, the font is easier to read.

**Same line:**

```
[{'degree_min': 1, 'degree_max': 111, 'degree_avg': 8.0, 'degree_stdev': 6.0}]
```

**Multiple lines:**

```
[{'degree_min': 1,
'degree_max': 111,
'degree_avg': 8.0,
'degree_stdev': 6.0}]
```

**DataFrame:**

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/comic2hero_df.png?raw=true'>

## Added Comic to Comic

I threw in the Comic to Comic weighted relationships mostly because I could. It does give me a second sample to examine without having to create another multigraph of graph models from a new dataset.

# Up Next

Global Counts Comparison

# Resources

- [Entry G4: Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/)
- [Entry G10: Local Metrics](https://julielinx.github.io/blog/g10_local_metrics/)
- [Entry G12: Degree Comparison](https://julielinx.github.io/blog/g12_degree_comparison/)
