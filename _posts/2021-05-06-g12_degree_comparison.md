---
title: "Entry G12: Degree Comparison"
categories:
  - Blog 
tags:
  - graph
  - neo4j
  - graph analytics
---

This is essentially a redo of the unweighted degree metrics from [Entry G10](https://julielinx.github.io/blog/g10_local_metrics/). I ran the same metrics and queries from that entry, except I used the multigraph as created in [Entry G11](https://julielinx.github.io/blog/g11_create_multigraphdb_desktop/). The switch to the multigraph allowed me to compare metrics across graph models. This will be a crucial aspect once I start comparing runtimes of the different queries.

I did still end up creating three notebooks. However, the results of these notebooks were much more enlightening than the previous set of three notebooks.

- [G12a notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/12a_nb_degree_comparison.ipynb) has the results grouped by function and Graph Model, the same way I did in the G10 notebooks
  - Example: there is a dataframe from Hero to Comic, Comic to Hero, and Undirected (all) for the size function for just the Bimodal Graph Model
  - Purpose: Shows different degree counts within the same Graph Model by function
- [G12b notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/12b_nb_degree_comparison.ipynb) has the results grouped by relationship type and Graph Model
  - Example: there is a dataframe for all the functions that pull Hero to Comic information for just the Bimodal Graph Model
  - Purpose: Compares different ways to pull the relationship type within the same Graph Model
- [G12c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/12c_nb_degree_comparison.ipynb) has the results grouped by relationship type
  - Example: there is a dataframe for the Hero to Comic relationships in both the Bimodal Graph Model and Mixed Graph Model
  - Purpose: Compares the results for a relationship type across Graph Models
  - Note: For this notebook, I only included one method of pulling the information; the size function or a straight pattern match
  
In addition to switching to the multigraph, I made several other changes. These changes allowed me to discover some inconsistencies as well as hone what I was looking at. These changes included:

- Included `OPTIONAL MATCH` instead of a full pattern match
- Used the `size` function as a method to pull metrics
- Put most of the results into DataFrames
- Alterned the `gsd` functions

## Included `OPTIONAL MATCH`

As discussed in [Entry 10](https://julielinx.github.io/blog/g10_local_metrics/) one of the first changes I made was to include an `OPTIONAL MATCH` statement in the pattern match query instead of a simple full `MATCH` statement:

```
MATCH (c1)
OPTIONAL MATCH (c1)-[]-(c2)
```

vs

`MATCH (c1)-[]-(c2)`

While running the `OPTIONAL MATCH` queries I decided to double check the bimodal model's pattern match of outgoing and incoming relationships. To do this I played with leaving the label off vs including it.

```
MATCH (c1)
OPTIONAL MATCH (c1)-[]->(c2)
```

vs

```
MATCH (c1:Hero)
OPTIONAL MATCH (c1)-[]->(c2)
```

While the minimum and maximum results were the same, the average and standard deviation were quite different. If we think through the cause, it becomes apparent why this is true. In using a directed relationship we can only traverse from `Hero` nodes to `Comic` nodes and not the reverse. This means that all `Comic` nodes return a degree of zero.

So, while the pattern I specified would only return results for Hero nodes connecting to Comic nodes (`(c1)-[]->(c2)`) the query still counts the Comic nodes, they just don't have any connections.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/directed_rel.png?raw=true'>

```
pd.DataFrame(bi_session.run('''MATCH (c1)
OPTIONAL MATCH (c1)-[]->(c2)
WITH c1, count(distinct c2) as degree
RETURN min(degree) as degree_min,
max(degree) as degree_max,
round(avg(degree), 2) as degree_avg,
round(stDev(degree), 2) as degree_stdev
''').data()).transpose().rename(columns={0:'incorrect'}).merge(
pd.DataFrame(bi_session.run('''MATCH (c1:Hero)
OPTIONAL MATCH (c1)-[]->(c2)
WITH c1, count(distinct c2) as degree
RETURN min(degree) as degree_min,
max(degree) as degree_max,
round(avg(degree), 2) as degree_avg,
round(stDev(degree), 2) as degree_stdev
''').data()).transpose().rename(columns={0:'correct'}), left_index=True, right_index=True)
```

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/pattern_match_discrepancy.png?raw=true'>

## Use the `size` Function

The second change I made was to include the `size` function as a way to pull the degree. This function is actually really easy to use. Once I incorporated what I learned from the `OPTIONAL MATCH` statement (use the appropriate label), it came together flawlessly and without surprises.

## Put Results in DataFrames

Putting the results into DataFrames was really helpful. In the [G12a notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/12a_nb_degree_comparison.ipynb) it was easy to see how the summary statistics changed based on what type of relationship I was looking at.

In the [G12b notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/12b_nb_degree_comparison.ipynb) I was able to see discrepancies across functions and/or Graph Models. For example, there are a different number of `KNOWS` relationships in the unimodal model and the mixed model, which shouldn't be the case; these models should have the exact same number of `KNOWS` relationships.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_degree_discrepancy.png?raw=true'>

*Note*, while completing [Entry G13](https://julielinx.github.io/blog/g13_weighted_degree_comparison/) it occurred to me that there may have been an error while loading the data. I cleared out the Mixed Model graph and reloaded the data. This fixed the discrepancy, which is reflected in the notebook.

This view of the data also forced me to hone exactly what I was looking at. For example, in the [G12a notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/12a_nb_degree_comparison.ipynb) I could call the results of the `gds` functions things like "NATURAL", "REVERSE", and "UNDIRECTED", but it wasn't clear how that lined up against the other functions. Lining up the results of the `gds` function allowed me to identify a problem with the results in the Bimodal Graph Model.

## `gds` function

The `gds` function is different from the other methods that get the degree. The major difference is that it has multiple required parameters. These parameters include:

- Node projection
- Relationship projection
  - Relationship type
  - Relationship orientation

These parameters allow the specification of what nodes and relationships (or projections thereof) to use.

- `nodeProjection` accepts node labels and other value clauses that would work in the pattern match
  - Examples:
    - label:Hero
    - value: {name: 'Steve Rogers'}
- `relationshipProjection` accepts relationship types and orientation
  - Examples:
    - type: 'KNOWS'
    - orientation: UNDIRECTED

I'm going to stop for a second on relationship orientation. Understanding these options will help you understand the problem I ran into. Orientation options:

- `NATURAL`
  - Follows the indicated relationship direction
  - Example: `(:Hero)-[:APPEARS_IN]->(:Comic)`
  - For the bimodal model, this will tell us the number of comics that a hero appears in
- `REVERSE` 
  - Reverses the indicated relationship direction
  - Example: `(:Hero)<-[:APPEARS_IN]-(:Comic)`
  - For the bimodal model, this will tell us the number of heroes in any given comic
- `UNDIRECTED`
  - Uses relationship regardless of direction
  - Example: `(:Hero)-[:APPEARS_IN]-(:Comic)`
  - This will include both the number of comics that a hero appears in and the number of heroes in any given comic

In looking at the examples, they should look suspiciously familiar from the `OPTIONAL MATCH` section above where I got inconsistent results. What we're dealing with is the same phenomenon: we're excluding the relationships we're not interested in, but not the start nodes.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/directed_rel.png?raw=true'>

I thought for sure I could exclude the nodes I wasn't interested in by using the `nodeProjection` parameter. However, this parameter only includes labels. So if I leave the `Comic` label out of the Bimodal Graph Model's `nodeProjection` then none of the nodes have any relationships. This is because the `Hero` nodes are only connected to the `Comic` nodes. By excluding them, we essentially isolate all of our `Hero` nodes.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/gds_nodeProjection.png?raw=true'>

Thus we're required to include both labels, which gives us the incorrect mean and standard deviation for the same reason the `OPTIONAL MATCH` did when we didn't include the starting node label.

If there is a way to do it, the documentation needs to be updated to be more human readable.

## Up Next

[Weighted Degree Comparison](https://julielinx.github.io/blog/g13_weighted_degree_comparison/)

## Resources

- [Degree Centrality](https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality/)
- [Analysis of commonly used together ingredients](https://guides.neo4j.com/4.0-intro-graph-algos-exercises/PracticalApplication.html?_gl=1*9if49j*_ga*MTQ0Mjk1MzQ0LjE2MTY1MTc2MDg.*_ga_DL38Q8KGQC*MTYxNzAzNDYzOC4zLjEuMTYxNzAzNDg1My4w&_ga=2.196961967.2064669514.1617034639-144295344.1616517608&_gac=1.175427094.1616517610.EAIaIQobChMI6qHOq-3G7wIVj7t3Ch0pWgOXEAAYASAAEgLfX_D_BwE)
- [Native projection](https://neo4j.com/docs/graph-data-science/current/management-ops/native-projection/#native-projection-syntax-relationship-projections)
- [Entry G10](https://julielinx.github.io/blog/g10_local_metrics/)