---
title: "Entry G8: Components"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

Now that I have a general feel for the graph database with counts and density, I want to look at components.

The notebooks where I did my code for this entry can be found on my github page. I created three notebooks, one for each graph model. These notebooks contain the code for Entries G6, G7, and G8.

- [Entries G6, G7, G8: Global Metrics Unimodal Graph Model](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8a_nb_unimodal_global_metrics.ipynb)
- [Entries G6, G7, G8: Global Metrics Biimodal Graph Model](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8b_nb_bimodal_global_metrics.ipynb)
- [Entries G6, G7, G8: Global Metrics Mixed Graph Model](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8c_nb_mixed_global_metrics.ipynb)

## Components

A component is just a set of connected nodes. I'll be looking at three measures:

 - Component count
 - Component size
 - Component percent 
 
Some quick facts about components:

- A single graph can have many components, although there is usually a largest component containing over 90% of the graph (page 305 of [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098) has a great chart showing basic metrics for 27 different graphs from 4 different industries)
- There is by definition no path between any pair of nodes in different components
- There are strongly connected components (i.e, cliques, which I talked about in [Entry G5](https://julielinx.github.io/blog/g05_project_bimodal/)) and weakly connected components (any single path between nodes)

Here's an example of a graph with two components from the simple bimodal example from [Entry G3](https://julielinx.github.io/blog/g03_graph_model/):

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/bimodal_components.png?raw=true'>

There is also something called k-components. This is the same concept as a regular component, but per page 180 of *Networks* "a set of nodes such that each is reachable from each of the others by at least k node-independent paths."

Here's an example from a subset of the Marvel Universe Social Network:

- Blue = 1-component
- Purple = 5-component
- Green = 2-component
- Red = 3-component

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/k-components.png?raw=true'>

### Component count

For this measure, all I want to know is how many components are in the graph.

The Marvel Universe Social Network has 22 components. However, 18 of those are isolated nodes. The question then becomes, do I want to count isolates as their own component?

My first thought was that the isolates are already accounted for *as* isolates in one of the earlier metrics. And this is 100% true for the unimodal model of the Marvel Universe Social Network. However, when looking at the metrics for the bimodal model of the graph, the 18 isolates are no longer isolated because those heroes are connected to the comic they appear in. As such, I'd need to include them in the components.

**Decision**: Isolates show nodes that aren't connected to any other nodes. As such, I can safely remove components with a size of 1 from the results without losing information.

### Component size

This is just the number of nodes in each component. For the unimodal model of the Universe Universe Social Network, there were four components (leaving out the 18 isolates as decided above).

As you can see in the table, this graph has the common giant component as referenced in the beginning of the entry.

<table align=left>
    <tr>
        <td>Component ID</td>
        <td>Node Count</td>
    </tr>
    <tr>
        <td>0</td>
        <td>6403</td>
    </tr>
    <tr>
        <td>239</td>
        <td>9</td>
    </tr>
    <tr>
        <td>92</td>
        <td>7</td>
    </tr>
    <tr>
        <td>3504</td>
        <td>2</td>
    </tr>
</table>

### Component percent

This measure takes the component size one step further. We take the number of nodes in each component and divide by the the number of nodes in the graph. This gives us the size of each component within the full graph.

For the four components of the Marvel Universe Social Network unimodal model, we end up with this:

<table align=left>
    <tr>
        <td>Component ID</td>
        <td>Node Count</td>
        <td>Node Percent</td>
    </tr>
    <tr>
        <td>0</td>
        <td>6,403</td>
        <td>99.44%</td>
    </tr>
    <tr>
        <td>239</td>
        <td>9</td>
        <td>0.14%</td>
    </tr>
    <tr>
        <td>92</td>
        <td>7</td>
        <td>0.11%</td>
    </tr>
    <tr>
        <td>3504</td>
        <td>2</td>
        <td>0.03%</td>
    </tr>
</table>

## Next Up

Local Metrics

## Resources

- [Neo4j Python Driver 4.2](https://neo4j.com/docs/api/python-driver/current/)
- [Fast counts using the count store](https://neo4j.com/developer/kb/fast-counts-using-the-count-store/)
- [Degree Centrality](https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality/)
- [Relationship Orientation](https://neo4j.com/docs/graph-data-science/current/management-ops/cypher-projection/#cypher-projection-relationship-orientation)
- [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098) by Mark Newman
- [Fraud Detection slideshare](https://www.slideshare.net/maxdemarzi/fraud-detection-and-neo4j) by Max De Marzi