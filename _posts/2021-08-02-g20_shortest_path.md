---
title: "Entry G20: Shortest Path"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

In an unweighted graph, the fewest steps between any node (A) to another node (B) is called the shortest path.

## The Problem

Let's look at an example. Say you want to drive from Cheyenne, WY to Fishlake National Park in Utah. There are three main routes:

- Northern route on I-80, passing through southern Wyoming and down into Salt Lake City, UT (blue)
- Middle route on US-40, passing through Medicine Bow-Routt National Forest and Vernal, UT (pink)
- Southern route on I-70, passing through Denver, CO and Grand Junction, CO (green)

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/20_cheyenne_denver_fishlake_routes.png?raw=true'>

If we represent this as a graph, we'll have cities as the nodes and roads as the connections.

In our Cheyenne to Fishlake example we'd have:

- Northern route: (Cheyenne) -[]-> (Laramie) -[]-> (Rawlins) -[]-> (Rock Springs) -[]-> (Evanston) -[]-> (Salt Lake) -[]-> (Provo) -[]-> (Fishlake)
- Middle route: (Cheyenne) -[]-> (Laramie) -[]-> (Vernal) -[]-> (Price) -[]-> (Fishlake)
- Southern route: (Cheyenne) -[]-> (Fort Collins) -[]-> (Denver) -[]-> (Breckenridge) -[]-> (Grand Junction) -[]-> (Green River) -[]-> (Fishlake)

Based on this representation we have the following number of steps:

- Northern route: 7 steps
  - (Cheyenne) -[1]-> (Laramie) -[2]-> (Rawlins) -[3]-> (Rock Springs) -[4]-> (Evanston) -[5]-> (Salt Lake) -[6]-> (Provo) -[7]-> (Fishlake)
- Middle route: 4 steps
  - (Cheyenne) -[1]-> (Laramie) -[2]-> (Vernal) -[3]-> (Price) -[4]-> (Fishlake)
- Southern route: 6 steps
  - (Cheyenne) -[1]-> (Fort Collins) -[2]-> (Denver) -[3]-> (Breckenridge) -[4]-> (Grand Junction) -[5]-> (Green River) -[6]-> (Fishlake)

From the above, we can clearly see that the middle route is by far the shortest as far as the simplified representation of the number of cities we'd pass through.

### Weighted Relationships

Now, if we ask Google Maps to find us the shortest path, we can see two additional considerations: time and distance;

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/20_cheyenne_fishlake_distances.png?raw=true'>

As you can see, Google has routed our southern route through Medicine Bow-Routt National Forest, probably due to the three accidents on I-70 west of Breckenridge. But let's go with it.

If we use distance in miles as our "shortest" criteria, the middle route is still our clear winner with 561 miles (50 miles shorter than the northern route and 21 miles shorter than the southern route). However, if we use time as our "shortest" criteria then the northern route wins, even though it is the farthest in miles and had the most steps in our original representation.

To add time or distance into the graph you include a "weight" on the relationship. So the relationship between Cheyenne and Laramie would look something like this:

(Cheyenne) -[**distance**: 51.3, **time**: 50]-> (Laramie)

Where "distance" and "time" are two different types of weights.

## The Options

Most of what I'm looking at in this series of posts are graph wide type of metrics, not specific information on one node to another. So, however we do shortest path, I'm going to want to apply it to every person node in the graph.

Now, looking at a person connected to another person (or comic in the Bimodal Model) won't really tell us anything, either a person is connected (shortest distance = 1) or it's not (shortest distance = 0). So, for my purposes, I'm going to use shortest path to a villain.

Based on the results from the [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb), most nodes will have a villain connection within one or two steps. But let's see just how far you *have* to travel before encountering a villain. Are there even any that are more than two steps away? I also want to have this metric for future reference because other graphs may have longer distances to nodes of interest.

I considered two solutions for this:

- `gds.shortestPath.dijkstra`
- `apoc.path.spanningTree`

### `gds.shortestPath.dijkstra`

This [algorithm function](https://neo4j.com/docs/graph-data-science/current/algorithms/dijkstra-source-target/) is designed to find the shortest path between nodes. However, it also requires the specification of a projected graph, which I haven't taught myself how to do in Neo4j yet. I'll be covering this in a future post, as many algorithm functions in the Graph Data Science Library use it, but for now, it'd just be another thing I'd have to spend time on before I could wrap up this post.

If you remember from waaaay back in [Entry 2](https://julielinx.github.io/blog/02_define_process/), I decided to follow the advice of Jason Brownlee to spend as close to [no more than 5-15 hours from inception to presentation of the results on small projects](https://machinelearningmastery.com/self-study-machine-learning-projects/) (i.e. each blog post - and yes, I still regularly blow right past that 15 hour mark).

As such, the Dijkstra Shortest Path algorithm isn't really a contender for this post's solution.

### `apoc.path.spanningTree`

As I [said before](https://julielinx.github.io/blog/g19_neighborhood_node_cts/#apocpathspanningtree), this [algorithm function](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/) is really handy. All we have to do to make it work with this use case from the last one is to make two changes:

- Use the termination filter (`/`) in the `labelFilter` parameter
- Include `limit: 1` in the configuration map

If you're paying attention, you realize that the termination filter, which tells the query to stop once it's encountered a specific label, is superfluous once we add the `limit: 1` condition. But this is a good time to practice using different label filters.

## The Solution

Running it on the Unimodal Model, the query changed from this:

```
MATCH (h:Hero)
call apoc.path.spanningTree(h, {labelFilter:'>Villain', minLevel: 1, maxLevel:6})
YIELD path
RETURN h.name as name, labels(h)[-1] as type, length(path) as distance, count(path) as villain_ct
```

to this:

```
MATCH (h:Hero)
call apoc.path.spanningTree(h, {labelFilter:'/Villain', minLevel: 1, limit: 1})
YIELD path
RETURN h.name as name, labels(h)[-1] as type, length(path) as min_distance
```

Then I looked for the maximum value. There were 9 people with a maximum distance of 3:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/20_villain_shortest_path.png?raw=true'>

## Next Up

[Diameter](https://julielinx.github.io/blog/g21_diameter/)

## Resources

- [Dijkstra Shortest Path algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/dijkstra-source-target/)
- [Expand Spanning Tree](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/)
- [Entry G19c notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/19c_nb_neighborhood_cts.ipynb)
- [Expanding Spanning Tree: Label Filters](https://neo4j.com/labs/apoc/4.1/graph-querying/expand-spanning-tree/#expand-spanning-tree-label-filters)
- [Entry 2: Define the Process](https://julielinx.github.io/blog/02_define_process/)
- [4 Self-Study Machine Learning Projects](https://machinelearningmastery.com/self-study-machine-learning-projects/)
- [entry G19: Neighborhood Node Counts](https://julielinx.github.io/blog/g19_neighborhood_node_cts/#apocpathspanningtree)
