---
title: "Entry G22: Mean Distance Between Connected Nodes"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

The `apoc.path.expandingTree` algorithm in [Entry G19](https://julielinx.github.io/blog/g19_neighborhood_node_cts/) revealed a gold mine of information. Once I had that table of results I knew that not only could I grab the diameter and shortest paths to villains, I could also use it to calculate the mean distance between pairs of nodes, which I hadn't even considered attempting.

[Math Insight](https://mathinsight.org/definition/network_mean_path_length) has a nice, succinct definition of this metric: "The mean path length is the shortest path length, averaged over all pairs of nodes."

## The Problem

You may or may not remember in [Entry G18](https://julielinx.github.io/blog/g18_egocentric_networks/) when I [referenced](https://julielinx.github.io/blog/g18_egocentric_networks/#step-distance) the mean distance between connected node pairs from Mark Newman's book [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098). This seems like a handy piece of information to have - I think by now we've all caught on to the fact that [I love summary statistics](https://julielinx.github.io/blog/g10_local_metrics/#degree-summary-statistics).

We already saw the distribution of distance between connected pairs in [Entry G19](https://julielinx.github.io/blog/g19_neighborhood_node_cts/), now it's time to get our single number summary of that data.

## The Solution

The beautiful table created in the [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_node_cts.ipynb) gives us all the information we need to calculate the mean number of steps.

The equation for mean distance between node pairs is:

avg_dist = $\frac{1}{n(n-1)} \times \sum d(v_{i}, v_{j})$

or as I calculate it (let's not over complicate things with multiplying by reciprocals instead of just dividing by the actual value):

avg_dist = $\frac{\sum d(v_{i}, v_{j})}{n(n-1)}$

Where:
- $n$ is the number of nodes
- $d(v_{i}, v_{j})$ is the step distance from node $v_{i}$ to node $v_{j}$

This is not as complicated as it looks. It breaks down to this: to get a mean (i.e., average) we need to sum all the things ($\sum d(v_{i}, v_{j})$), then divide by the count of all the things ($n(n-1)$).

To accomplish this from the table of results we got from the [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_node_cts.ipynb) we need three steps to sum all the things, then we need to figure out exactly what it is we're dividing by:

- Sum all the things
  1. Sum the number of paths at each step level
  2. Multiply the number of paths by the distance
  3. Sum the results to get the total number of steps
- Divide by the count of all things

### Sum All The Things

First we need to know how many paths there are at each step level. Because we used a breadth first based algorithm, we have the shortest paths between all connected pairs of nodes. If we group by distance and sum the counts, we get the total number of paths at each step level:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/22_distance_sums.png?raw=true'>

We can't very well just add all these together because they represent different numbers of steps. To get the number of steps at each distance, we need to multiply the sum of paths by the step distance:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/22_distance_sums_mult.png?raw=true'>

Now we just sum the `total_nbr_steps` column and voilà, we now have the total number of steps in our graph, which is 107,821,826.

### Divide by The Count of All Things

It's tempting to think that the "count of things" is the number of nodes. However, we're not counting nodes, we're counting paths, so we need the combination of every node with every other node. This is the $n(n-1)$ portion of the equation. You may recognize it from the calculation for the number of possible relationships in [Entry G7](https://julielinx.github.io/blog/g07_global_density_diameter/#number-of-possible-relationships).

*Note*: Using this equation, a node can connect to every node except itself - which brings up a set of constraints: we're assuming that a node cannot be connected to itself and that it connects a maximum of one time to any other node.

If we want to double-check this thinking, we can calculate the mean distance between connected nodes using a little algebra where every node is connected to every other node.

The mathematical equivalent of "every node being connected to every other node" is $n(n-1)$. Using this we come up with the following equation:

$\frac{n(n-1)}{x} = 1$

Using a little algebraic magic we get this:

$=> \frac{1}{n(n-1)} \times \frac{n(n-1)}{x} = 1 \times \frac{1}{n(n-1)}$

$=> \frac{1}{x} = \frac{1}{n(n-1)}$

$=> x = n(n-1)$

Of course, you could also just know that any number divided by itself is 1 and save yourself some time.

Plugging in our numbers, we end up with 6,421 $\times$ 6,420 = 41,222,820

### Get the Answer

Now all we have to do is plug in our numbers:

$\frac{107821826}{41222820} = 2.62$

This value makes sense based on the charts from [Entry G19](https://julielinx.github.io/blog/g19_neighborhood_node_cts/) and the counts of paths at each level - distances of three and two were most popular.

## Up Next

Nearest Neighbor Egonet Density

## Resources

- [Mean path length definition](https://mathinsight.org/definition/network_mean_path_length)
- [Average path length](https://en.wikipedia.org/wiki/Average_path_length)
- [Entry G7: Global Density and Diameter](https://julielinx.github.io/blog/g07_global_density_diameter)
- [Entry G19: Neighborhood Node Counts](https://julielinx.github.io/blog/g19_neighborhood_node_cts/)
