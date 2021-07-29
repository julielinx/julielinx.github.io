---
title: "Entry G18: Egocentric Networks"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

Now that we've gone through the global metrics and have a feel for the structure and composition of our data, it's time to start running local metrics.

There is a very handy construct to help us formulate local metrics: the egocentric network.

## Context

Up to this point we've been looking at the graph as a whole.

A very cleaned up version of the Marvel graph looks like this example from Tomaz Bratanic's post [GRAPH ALGORITHMS, MARVEL, NEO4J: Neo4j Marvel Social Graph Algorithms Community Detection](https://tbgraph.wordpress.com/2017/11/17/neo4j-marvel-social-graph-algorithms-community-detection/):

<img src='https://tbgraph.files.wordpress.com/2017/11/marvel_social_louvain.png?w=768'>

However, there was a lot of cleaning and noise filtering that went into creating that image.

To give you an idea as to why we can't always look at the whole graph, here's just the 6,439 Heroes (our unimodal graph model) and their 171,644 relationships with minimal cleaning and noise filtering:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/marvel_network.png?raw=true'>

Pretty, but it's quite messy and difficult to parse any data from this view. To get this image, I used the APOC library to export the nodes and relationships to Gephi. For directions on how to do this see Tomaz Bratanic's post [Neo4j to Gephi](https://tbgraph.wordpress.com/2017/04/01/neo4j-to-gephi/) on his blog [Graph People](https://tbgraph.wordpress.com/) (also a wonderful source of other graph related content).

Another reason we might not want to look at the graph as a whole is if we're trying to make predictions on new information as it comes in. In this case, we'd have to run our global metrics against every new piece of information as it came in, and the changes from a single addition probably won't be informative.

## Egonet

To get a more granular view, we can look at a subset of nodes centered around a specific node of interest (or in my case, every node one at a time). In other words, we pull up a single node and look at the nodes around it.

A small piece of a graph is called a **subgraph**. This simply means we're looking at a subset of the full graph. Subgraphs aren't limited to a local neighborhood as we'll be exploring, it can be any subset of nodes and their relationships.

### Nearest Neighbors

A specific node and the nodes it's directly connected to would look like this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/egonet_simple.png?raw=true'>

These directly connected nodes are called nearest neighbors. Another way to reference them is to say they are one step from the node of interest or at a distance of 1.

Our subset becomes more complex, but also more informative, as we include the relationships between the nearest neighbors. In the image below, there are two extra relationships: one between two of the nearest neighbors on the left and one between two other nearest neighbors at the upper right.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/egonet_simple_connected.png?raw=true'>

### Next Nearest Neighbors

We can expand this to include the next layer of nodes, which are called the next nearest neighbors.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/egonet_expanded_simple.png?raw=true'>

## Caveats

As we expand outward and add more steps to the next node and more relationships there are several issues we need to watch out for.

### Multiple Paths

Returning to our nearest neighbor example, if we follow certain paths, we can actually turn our nearest neighbors ("H1" - distance of one step) into next nearest neighbors ("H2" - distance of two steps).

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/egonet_simple_conn_caveat.png?raw=true'>

### Breadth First Search (BFS) Solution

Using "breadth first search" solves this problem. Breadth first search expands to the nearest neighbors first, then to the nearest neighbors of those nodes without ever revisiting a node. It continues to do this until it runs out of neighbors to visit or hits some user specified limit. The most important aspect of this for our uses in analysis is that breadth first search never visits the same nodes twice.

We can see how useful this is when we look at a representation of a next nearest neighbor graph that shows the relationships between all nodes:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/egonet_expanded_connected.png?raw=true'>

As we expand outward, the complexity continues to increase. Breadth first search gives us a way to control that complexity and ensure we're returning the results we expect.

## Step Distance

Now that we know about egocentric networks, nearest neighbors, next nearest neighbors, step distance, and breadth first search, the question becomes: how far out do we expand?

On page 305 of [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098) Mark Newman provides a list of the mean distance between connected nodes pairs for most of the 27 networks he analyzed. Of the 23 networks with mean distance, the overall average was 6.5. If you remember from a previous post (or bought and read the book), Mark analyzed four sectors: Social, Information, Technology, and Biological. At a more granular level it broke down as follows:

<table>
    <tr>
        <th>Sector</th>
        <th>Mean</th>
        <th>Minimum</th>
        <th>Maximum</th>
        <th>Number of Networks</th>
    </tr>
    <tr>
        <td>Social</td>
        <td>6.62</td>
        <td>3.48</td>
        <td>16.01</td>
        <td>8</td>
    </tr>
    <tr>
        <td>Infomation</td>
        <td>10.77</td>
        <td>4.86</td>
        <td>16.18</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Technology</td>
        <td>6.80</td>
        <td>2.16</td>
        <td>18.99</td>
        <td>7</td>
    </tr>
    <tr>
        <td>Biological</td>
        <td>3.46</td>
        <td>1.90</td>
        <td>6.80</td>
        <td>5</td>
    </tr>
</table>

*Side note*: the full list, as well as the mean calculations, are in the accompanying [Entry G18 notebook](http://localhost:8888/lab/tree/datascience_diaries/graph/18_egonet.ipynb).

When working with the data at my job, I found that going three steps or more had diminishing returns.

There is a reference in one of the books I'm reading that discusses the diminishing influence within a social network at increasing distances, but I can't remember which book or at what step level they found the influence diminished significantly. I think it was at step three (the level past next nearest neighbors) or four, but I'm not sure. I'll continue to look for the reference and update the post if I find it. It was probably in either [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098), [Understanding Dark Networks](https://www.amazon.com/Understanding-Dark-Networks-Daniel-Cunningham/dp/1442249447), or [A First Course in Network Science](https://www.amazon.com/First-Course-Network-Science/dp/1108471137). 

In the meantime, I'll work out some fast running metrics indicative of distance usefulness, mostly using shortest path (which will be coming up in one of the next posts). I assume the ideal distance depends on the specific graph.

For the most part, I'll limit the egonets to next nearest neighbors simply because of limitations around run time and memory usage.

## Up Next

Neighborhood Node Counts

## Resources

- [GRAPH ALGORITHMS, MARVEL, NEO4J: Neo4j Marvel Social Graph Algorithms Community Detection](https://tbgraph.wordpress.com/2017/11/17/neo4j-marvel-social-graph-algorithms-community-detection/)
- [Neo4j to Gephi](https://tbgraph.wordpress.com/2017/04/01/neo4j-to-gephi/)
- [Graph People](https://tbgraph.wordpress.com/)
- [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098)
- [Understanding Dark Networks](https://www.amazon.com/Understanding-Dark-Networks-Daniel-Cunningham/dp/1442249447)
- [A First Course in Network Science](https://www.amazon.com/First-Course-Network-Science/dp/1108471137)
