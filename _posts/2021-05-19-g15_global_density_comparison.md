---
title: "Entry G15: Global Density Comparison"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

The [notebook that accompanies this entry](https://github.com/julielinx/datascience_diaries/blob/master/graph/15_nb_global_density_comparison.ipynb) is a cleaned up, concise version of the three notebooks I created for [Entry G7](https://julielinx.github.io/blog/g07_global_density_diameter/), addressing only the number of possible relationships and density. Just like [Entry G14](https://julielinx.github.io/blog/g014_global_counts_comparison/), this is a supplement to, not a replacement for, the older entry ([G7](https://julielinx.github.io/blog/g07_global_density_diameter/)) so make sure you read the older one first.

## Possible Relationships

The mathematical equations are in [Entry G7](https://julielinx.github.io/blog/g07_global_density_diameter/). I won't repeat that here, but I thought some visualizations may help with intuitive understanding of why we calculated the unimodal vs the bimodal graph models differently.

The difference in the equations is entirely to do with what type of node can connect to what other type of node.

In the Unimodal Model, any node can connect to any other node as shown in the picture below:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/unimodal_possible_rels.png?raw=true'>

We can also connect any node to any other node in the Mixed Model. The difference is that the comic nodes are in that Model as well as the Hero nodes.

When we get to the Bimodal Model however, there are limitations around what type of node can connect to what type of node. For example, you can't connect a Comic node to another Comic node, it has to be connected to a Hero node. We can see this limitation here:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/bimodal_possible_rels.png?raw=true'>

When we look at the actual numbers we can really see how the number of nodes and the way they are allowed to connected effects the number of possible relationships.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/possible_rels.png?raw=true'>

In comparing the Unimodal Model to the Mixed Model (both using the same allowed connections of any node to any other node), we can see how the number of nodes effect the possible relationships. We jump from 20,727,141 possible relationships in the Unimodal Model to 182,204,505 possible relationships in the Mixed Model. This is a significant increase (over 160 million) especially considering we only added 12,651 nodes.

In comparing the Bimodal Model and the Mixed Model (which both have the same number of nodes), we can see the effect of allowed connections. We go from 81,459,789 possible relationships in the Bimodal Model to 182,204,505 possible relationships in the Mixed Model. At just over 100 million, this increase is less than the difference between Unimodal and Mixed Modes, but still significant.

*Side note*, now that I'm thinking about it and creating pictures, *technically* the number of possible relationships in the Mixed Model would be the possible number in the Unimodal Model added to the possible number in the Bimodal Model because we still don't connect Comic nodes to other Comic nodes in the Mixed Model.

That would be 102,186,930 possible relationships. Keep in mind that the more types of nodes you have and the more defined the model of which nodes can connected to which other nodes, the more complex your calculation will be for the number of possible nodes.

However, at the moment, I find the simplified calculation more informative due to the ease of comparison with the other graph models. 

## Density

Once we've got a handle on the number of possible relationships (and more importantly, how that number can change depending on our graph model), density is easy to understand. It's just the number of actual relationships divided by the number of possible relationships.

As indicated in [Entry G7](https://julielinx.github.io/blog/g07_global_density_diameter/), this value always lies between 0 and 1 (unless you model your relationships as a multigraph - i.e. you allow more than one connection between any two nodes. For more on relationships in a multigraph model refer back to [Entry G4](https://julielinx.github.io/blog/g04_graph_model_rels/)).

### Dense

A graph with a density of 1 would have every node connected to every other node and would look like this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/clique2.png?raw=true'>

### Sparse

A more sparse graph might look something like this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/clique_made_sparse.png?raw=true'>

### Isolated

A graph with a density of 0 would have no relationships and would look like this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/clique_opposite.png?raw=true'>

### Comparison

Now we can compare the actual densities of our three graph models. Note that all of our models have less than 1% of the relationships that they *could* have. This makes all of them sparse networks. As discussed in [Entry 10](https://julielinx.github.io/blog/g10_local_metrics/) most real world networks are sparse.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/marvel_density.png?raw=true'>

Unsurprisingly, the Unimodal Model is the most dense. It has the fewest nodes, which means it has the lowest number of possible relationships, and it has more actual relationships than the Bimodal Model. To make that small number a little easier to interpret, the Unimodal Model has 0.83% of the relationships that it could have.

The Bimodal and Mixed Models have very similar densities. Bimodal has 0.12% of the possible relationships it could have and Mixed has 0.15%.

*Note*, if we use the technically correct way to calculate the number of possible relationships for the Mixed Model (as discussed above), the density changes to 0.00262. This nearly doubles the density of that graph, but still leaves it significantly more sparse than the Unimodal Model.

It will be interesting to see if these generalities hold true for the other datasets I plan to do. I expect it will since the projection from bimodal to unimodal is the same regardless of the data.

Where I would expect divergence from this pattern would be if we modeled the unimodal graph based on specific criteria instead of just projecting it. For a reminder on some of the quirks of projecting bimodal data to a unimodal graph, see the [Cliques Caveat](https://julielinx.github.io/blog/g05_project_bimodal/#cliques-caveat) section of [Entry G5](https://julielinx.github.io/blog/g05_project_bimodal).

# Up Next

[Components Comparison](https://julielinx.github.io/blog/g16_components_comparison/)

# Resources

- [Entry G4: Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/)
- [Entry G7: Global Density and Diameter](https://julielinx.github.io/blog/g07_global_density_diameter/)
