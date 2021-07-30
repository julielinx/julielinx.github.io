---
title: "Entry G19: Neighborhood Node Counts"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

Now that we've established [how to define an egocentric neighborhood](https://julielinx.github.io/blog/g18_egocentric_networks) and [differentiated our people nodes](https://julielinx.github.io/blog/g17_add_villains), it's time to start calculating metrics for the different neighborhoods.

The notebooks where I did the code for this entry are:

- [Entry 19a notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19a_nb_neighborhood_cts.ipynb)
- [Entry 19b notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19b_nb_neighborhood_cts.ipynb)
- [Entry 19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb)

## The Problem

Nodes that are more closely connected tend to share similar characteristics. In the larger population, we can see this in accents within regions of same speaking countries. In the United States there are distinct accents in the south, the far north (like Minnesota), and Boston. The principles behind this are *assortativity* and *homophily*.

- Assortativity: "a preference for a network's nodes to attach to others that are similar in some way" [Assortativity - Wikipedia](https://en.wikipedia.org/wiki/Assortativity)
- Homophily: "the tendency of individuals to associate and bond with similar others" [Homophily - Wikipedia](https://en.wikipedia.org/wiki/Homophily)

These concepts are very similar. [A First Course in Network Science](https://www.amazon.com/First-Course-Network-Science/dp/1108471137) shares these examples on page 36:

- Assortativity: relatives that live near each other or friends who have similar interests
- Homophily: people who practice the same sport or hobby are more likely to meet and become friends

The easiest way to track this is to measure who is connected to who. To start this off I did simple counts: number of nearest neighbors, number of next nearest neighbors, and the percent of each that are villains.

Since we're counting people nodes (everything with a "Hero" label) and we're using the same base data (there are no more or fewer people in any graph models compared to the others), the counts should be the same for all three graph models: unimodal, bimodal, and mixed. I verify that this is true in [Entry 19b notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19b_nb_neighborhood_cts.ipynb) where I charted the distributions for all neighbors, villain neighbors, and villain neighbor percentages.

*Confession*: okay, okay, you caught me. I've been working with the relationships so much that it didn't even occur to me that the node counts would be the same across graph models. Which is why I created [Entry 19a notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19a_nb_neighborhood_cts.ipynb), where I pull and visualize each of the graph models and step levels separately for nearest neighbors and next nearest neighbors.

## The Options

There are several ways to count the neighborhood around a node. Some of the methods I considered included:

- Cypher statements
- `gds.shortestPath.dijkstra`
- `apoc.path.subgraphAll`
- `apoc.path.spanningTree`

### Cypher Statements

Direct cypher statements are always a good choice. They provide the most flexibility, but they also require optimization and a good working knowledge of cypher and network theory to ensure avoidance of pitfalls like those discussed in the Caveats section of [Entry G18](https://julielinx.github.io/blog/g18_egocentric_networks).

### Library Functions

Using the Graph Data Science or APOC libraries can provide guardrails for pitfalls while still offering sufficient flexibility and optimized runtimes. 

*Caution*: I do recommend verifying that the actual results and expected results of any pre-written function match. As discussed in [Entry G10: Local Metrics](https://julielinx.github.io/blog/g10_local_metrics/), you don't always get the results you expect and sometimes the parameters may not do what you think.

#### `gds.shortestPath.dijkstra`

Using a [shortest path algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/dijkstra-source-target/) was my first thought.

The big limitation of the Dijkstra shortest path algorithm is that you generally only return a single result. It is literally looking for the shortest path between Node A and Node B. These types of problems are very common in the logistics field: semi A needs to get its goods from the manufacturer to the distributor in as short a distance or as short a time as possible.

For this first node count metric, I decided I wanted to know about all the nodes at one and two steps.

#### `apoc.path.subgraphAll`

I love [this algorithm](https://neo4j.com/labs/apoc/4.0/overview/apoc.path/apoc.path.subgraphAll/) in principle. It lets me specify the distance, then return everything within that criteria. I end up using it in the density calculations, but the farther out you go, the slower the algorithm runs.

In the [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb) I went out 6 steps just for the fun of it. This would have taken forever using the `apoc.path.subgraphAll` function.

Another limitation for this use case is that the function returns a list. There isn't an easy way to specify the step distance or grab the node labels. I would have had to do a lot of subtraction counts to get the information I wanted.

I also needed labels for multiple purposes: I used the start node label to color the histograms to more easily see if villains are more likely to be connected to other villains and to get the villain vs total counts to have the villain percentages. 

#### `apoc.path.spanningTree`

On the advice of Neo4j's Andrew Bowman, I looked at the [apoc.path.spanningTree function](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/). This function is pretty versatile. The major benefit of this algorithm is the breadth first search functionality as discussed in the [Caveats](https://julielinx.github.io/blog/g18_egocentric_networks/#caveats) section of [Entry G18](https://julielinx.github.io/blog/g18_egocentric_networks). Breadth first search prevents the accidental double counting of unique nodes or incorrect step distance due to multiple paths leading to that node from the start node.

In the image below, we can see that there are three paths (through nodes Ha, Hb, and Hc) to get from H1 to H3:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_mult_paths.png?raw=true'>

This image (also in Entry G18), shows how the step distance can change depending on the path we take to get to a particular node:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/egonet_simple_conn_caveat.png?raw=true'>

Other benefits of this algorithm include:

- node filtering that allows for easily counting villains while still expanding past them to find villains at the next step level
- relationship filters that make transforming the query to work on the mix graph a cinch
- easy to use

## The Proposed Solution

Obviously from the paragraph above I chose `apoc.path.spanningTree`.

I used two statements:

- First: pull the villain counts at various step levels
- Second: pull total counts at the same step levels

The statement to count villains in the unimodal graph model looks something like this:

```
MATCH (h:Hero)
call apoc.path.spanningTree(h, {minLevel: 1, maxLevel:2, labelFilter:'>Villain'})
YIELD path
RETURN h.name as name, length(path) as distance, count(path) as villain_ct
```

The [documentation for the spanningTree algorithm](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/#expand-spanning-tree-config) has the full list of parameters, but here's how the query above breaks down, including what parameters I used and why: 

- `MATCH (h:Hero)` defines the start node - or in our case, the label for multiple start nodes
- `call apoc.path.spanningTree()` calls the spanningTree algorithm
- `h` inserts our start node into the algorithm
- The values inside `{}` are our configuration map parameters. Pretty much everything in here is optional as far as running the algorithm. However, they can be very handy to return exactly what we want and limit our resource use
  - `minLevel` allows us to set a minimum distance. I set it at 1 to remove the count for the start node (distance of 0)
  - `maxLevel` allows us to set a maximum distance. To start I kept it at 2 so that I would only be looking at nearest neighbors and next nearest neighbors
  - `labelFilter` this parameter allows us to include, exclude, terminate, or end the expansion based on the value provided - the full list of options and how to denote them are in the [documentation](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/#expand-spanning-tree-label-filters) <img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_end_points.png?raw=true'>
    - I used the `>` symbol to indicate I wanted to only end on nodes with a "Villain" label, but that I also wanted to extend past those nodes to count next nearest villain neighbors too
    - When counting all people nodes, I changed `>Villain` to `>Hero`, since we left the "Hero" label on all people nodes (see the next example below)
      - This means that if we want to count only actual Heroes (that aren't villains), we'd need to either take the difference between `total_ct` and `villain_ct` or we'd need to remove the "Hero" label from all "Villain" nodes
      - I find a label for all people nodes helpful, so if we were to change our labelling scheme, we might end up with three labels on people nodes: "Person", "Hero", and "Villain"
- `YIELD` returns the paths
  - Returning the path does present some limitations around what you can reference for further metric calculations
  - Example: we're running two statements (one for villains and one for everyone) to get separate counts, then merging them in pandas
  - If the return information was different, we could simply return the label for the end node then group counts by label
- The `RETURN` statement is where we specify what we actually want returned
  - Since we're using a label as the start node, we need `h.name` to break out the counts by each individual person node - you can think of this like using a GROUP BY type statement
  - `length(path)` tells us what step level we're at. Remember because of the `minLevel` and `maxLevel` parameters, we'll only return the nearest and next nearest neighbors (I did extend this out to farther distances in the [Entry 19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb), which I talk about later in the post)
  - `count(path)` tells us how many paths there are at a given step level
    - Because `apoc.path.spanningTree()` uses breadth first search, we don't have to worry about getting multiple paths to the same node. This allows us to use the path count as a stand in for the number of nodes at each step level

When moving to the Mixed Model, we need to add another parameter to the configuration map. Our query now looks something like this:

```
MATCH (h:Hero)
    call apoc.path.spanningTree(h, {minLevel: 1, maxLevel:2, labelFilter:'>Hero', relationshipFilter:'KNOWS'})
    YIELD path
    RETURN h.name as name, length(path) as distance, count(path) as total_ct
```

- `relationshipFilter` let's us follow only the relationships we specify
  - To replicate the Unimodal Graph Model, we set this parameter to 'KNOWS' so we're only moving between people nodes
  - To replicate the Bimodal Graph Model, we set this parameter to 'APPEARS_IN' so that we traverse through "Comic" nodes
- In this example I changed `>Villain` to `>Hero` to get the total people node count

For those of us that truly appreciate the fact that a picture is worth 1,000 words, here's what the results look like when you combine them into a single dataframe:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_result_dataframe.png?raw=true'>

If you're paying attention, you'll notice that what I have labelled as the villain percent is actual a ratio. Apparently I forgot to multiply by 100.

## The Results

I used stacked histograms to examine the data. I like histograms because they give me a feel for the distribution of the data. I also tend to have straight counts right next to a log scale version of the chart so I can more easily see lower count values.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_counts_and_scale.png?raw=true'>

There are a couple of things to keep in mind when looking at these charts:

- the villain color doesn't reflect anything about the connections. The coloring indicates the label of the start node: "Villain" or "Hero" where "Hero" really does mean non-villain, not every person node

- the coloring is misleading when looking at the y log scale. The bottom color seems much more prevalent, but that's because 0-10 takes up as much space as 100-1000 due to the log scale

When looking at the charts, I recommend the [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb) because I refind what I wanted to see and how to display it in the other two notebooks. This left notebook "c" the most condensed and information rich notebook.

### Assortatitivy, Homophily, and Villain Percent

The first set of graphs are organized by distance. This makes it easier to see the total count, villain count, and villain percent for any given distance.

Using the principles of assortativity and homophily, I would expect villains to have a higher percentage of villains - so the villain color would take up more and more space as we move toward 100%. However, when we look at a distance of 1, there is a block of non-villains with a villain percent around 60, and people nodes with a villain percent over 80 are all non-villains.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_dist1_villain_pct.png?raw=true'>

I attribute this to a quirk of using comics as our dataset: there is almost always at least one villain in a comic (but frequently not many more than one) and the number of non-villain characters usually out number the villains significantly. 

With the way the bimodal network is projected to a unimodal network, this means that pretty much every person node will be connected to a villain, as discussed in [Entry G5: Projecting Bimodal to Unimodal](https://julielinx.github.io/blog/g05_project_bimodal/). These facts inflate the number of villains that non-villains are connected to and decrease the number of villains that other villains are connected to.

When moving to other - more real world - datasets, I expect these results to be different.

### Step Level / Distance

Having run the counts out to a distance of 6 in the [Entry 19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb), we can start to see some interesting things about how an individual's network expands.

At a distance of 1 the vast majority of our nodes have less than 250 degrees, with the maximum being somewhere around 2,000. In statistics, they call this a "long tail."

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_distance1_node_cts.png?raw=true'>

When we move out to a distance of 2, the distribution is still weighted toward the middle and lower end for everyone, but it's much more evenly distributed across the board. The villains are almost evenly distributed across the whole range.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_distance2_node_cts.png?raw=true'>

At a distance of 3, we can see why influence decreases after a certain point: the distribution has reversed. There are now more people with a large number of connections than a small number of connections.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_distance3_node_cts.png?raw=true'>

However, as we continue out even farther, we see the number of connections decreasing. This is where that "mean distance between connected nodes" thing we were talking about in the [Step Distance](https://julielinx.github.io/blog/g18_egocentric_networks/#step-distance) section of [Entry G18](https://julielinx.github.io/blog/g18_egocentric_networks/) comes in to play, at a certain step distance there aren't any more nodes to connect to.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/19_distance5_node_cts.png?raw=true'>

If you go over to the [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_node_cts.ipynb), you'll notice that the charts don't go past distance 5. I thought this was a fluke, that `maxLevel` was one of those "up to but not including" type of things. So I increased `maxLevel` to 16. But the charts still only went up to a distance of 5.

Looks like I have my answer on the diameter of this particular graph, which is exciting. Guess I finally found a way to calculate diameter that doesn't max out the memory of my laptop. Sweet.

## Up Next

Mean Distance Between Connected Nodes

## Resources

- [Expand a spanning tree](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/)
- [APOC Documentation](https://neo4j.com/labs/apoc/4.1/)
- [Dijkstra Shortest Path](https://neo4j.com/docs/graph-data-science/current/algorithms/dijkstra-source-target/)
- [apoc.path.subgraphAll](https://neo4j.com/labs/apoc/4.0/overview/apoc.path/apoc.path.subgraphAll/)
- [Entry G17: Add Villains](https://julielinx.github.io/blog/g17_add_villains/)
- [Entry G18: Egocentric Networks](https://julielinx.github.io/blog/g18_egocentric_networks/)
- [Assortativity - Wikipedia](https://en.wikipedia.org/wiki/Assortativity)
- [Homophily - Wikipedia](https://en.wikipedia.org/wiki/Homophily)
- [A First Course in Network Science](https://www.amazon.com/First-Course-Network-Science/dp/1108471137)
- [Entry G10: Local Metrics](https://julielinx.github.io/blog/g10_local_metrics/)
- [Entry G5: Projecting Bimodal to Unimodal](https://julielinx.github.io/blog/g05_project_bimodal/)
