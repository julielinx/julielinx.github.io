---
title: "Entry G14: Global Counts Comparison"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

The [notebook that accompanies this entry](https://github.com/julielinx/datascience_diaries/blob/master/graph/14_nb_global_counts_comparison.ipynb) is the cleaned up, concise version of the three notebooks that accompanied [Entry G6](https://julielinx.github.io/blog/g06_global_counts/), but limited to just the global counts for the three graph models. 

As long as I'm cleaning things up, I decided to provide some additional pictures and commentary on global counts. Keep in mind that this entry is a supplement to [Entry G6](https://julielinx.github.io/blog/g06_global_counts/), not a replacement, so be sure to read that entry first.

## Overview

Now that the info for all three graph models are pulled into the same notebook, we can really start to see how the graph model effects the nodes and relationships in the graph.

The global count metrics I used can be summed up by counting the nodes and relationships in the picture below:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/global_counts.png?raw=true'>

The picture has:

- 16 nodes, of which 12 are Hero nodes and 4 are Comic nodes
- 13 relationships (all of which are between a Hero and a Comic, never between two of the same node type)
- 1 isolated Hero node (outlined in orange)

## Node Counts

With the node counts for all three graph models in the same DataFrame it's easy to see how the graph model effects the nodes in each graph.

The Hero nodes are the same for all three models and the Comic nodes are the same for the Bimodal Model and Mixed Model. As far as the nodes are concerned, the only different is that the Comic nodes were removed from the Unimodal Model.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hero</th>
      <th>Comic</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>uimodal</th>
      <td>6439</td>
      <td>0.0</td>
      <td>6439.0</td>
    </tr>
    <tr>
      <th>bimodal</th>
      <td>6439</td>
      <td>12651.0</td>
      <td>19090.0</td>
    </tr>
    <tr>
      <th>mixmodal</th>
      <td>6439</td>
      <td>12651.0</td>
      <td>19090.0</td>
    </tr>
  </tbody>
</table>
</div>

## Relationship Count

The relationships are where the main differences are between the three graph models.

When looking at the total relationship counts, each model has a different value:

- Bimodal is smallest with 96,104
- Unimodal is in the middle with 171,644
- Mixed is largest with 267,748

Keep in mind that the Unimodal Model has weighted relationships (for information on weighted relationships see [Entry G4](https://julielinx.github.io/blog/g04_graph_model_rels/)). While we are projecting relationships based on the original Bimodal Model, we end up with a lot more connections in the projected version. If we include the weights we get a total relationship count of 579,191 for Hero to Hero relationships (which we know from the [Entry G13 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/13a_nb_weighted_degree_comparison.ipynb)). That's around 6 times the number of connections from the original representation.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>KNOWS</th>
      <th>APPEARS_IN</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>unimodal</th>
      <td>171644.0</td>
      <td>0.0</td>
      <td>171644.0</td>
    </tr>
    <tr>
      <th>bimodal</th>
      <td>0.0</td>
      <td>96104.0</td>
      <td>96104.0</td>
    </tr>
    <tr>
      <th>mixmodal</th>
      <td>171644.0</td>
      <td>96104.0</td>
      <td>267748.0</td>
    </tr>
  </tbody>
</table>
</div>

When we break these down by relationship type, it becomes obvious that the Unimodal and Mixed Models have the same count for `KNOWS` relationships, while the Bimodal and Mixed Models have the same count for `APPEARS_IN`. The total count for the Mixed Model is just the addition of the relationships in the Unimodal and Bimodal Models.

This reflects how we created the models:

1. Started with Bimodal:
  - Hero nodes
  - Comic nodes
  - `APPEARS_IN` relationships
2. Created the Mixed model: added the `KNOWS` relationship between Hero nodes
  - Hero nodes
  - Comic nodes
  - `APPEARS_IN` relationships
  - `KNOWS` relationships
3. Reduced to the Unimodal model: removed the Comic nodes and all the `APPEARS_IN` relationships
  - Hero nodes
  - `KNOWS` relationships

## Isolate Count and Percent

The isolate count and percent just give us more information about the connectedness of the graph. When we look at the values for the Bimodal Model below, we can see that the diagram used to illustrate the three global counts at the beginning of the entry is actually impossible.

That graph has two nodes types (must be the Bimodal or Mixed Models) and relationships only between Hero to Comic (rules out the Mixed Model). However, the actual metrics tell us that there are no isolated nodes in the Bimodal Model. Good thing that diagram was for illustration purposes only.

As noted in [Entry G6](https://julielinx.github.io/blog/g06_global_counts/) the only graph model that has isolated nodes is the Unimodal Model. Even this graph has a very small percent of isolated nodes at 0.28%.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_count</th>
      <th>relation_ct</th>
      <th>isolates_count</th>
      <th>isolates_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>unimodal</th>
      <td>6439</td>
      <td>171644</td>
      <td>18</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>bimodal</th>
      <td>19090</td>
      <td>96104</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>mixmodal</th>
      <td>19090</td>
      <td>267748</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

# Up Next

[Global Density Comparison](https://julielinx.github.io/blog/g15_global_density_comparison)

# Resources

- [Entry G4: Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/)
- [Entry G6: Blobal Graph Counts](https://julielinx.github.io/blog/g06_global_counts/)
- [Entry G13: Weighted Degree Comparison](https://julielinx.github.io/blog/g13_weighted_degree_comparison/)
- [Entry G13 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/13a_nb_weighted_degree_comparison.ipynb)
