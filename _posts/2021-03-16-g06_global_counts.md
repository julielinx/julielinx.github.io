---
title: "Entry G6: Global Graph Counts"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

The first thing I want to do is get some global measures on the overall graph. I'm breaking these up into three categories: counts, density and diameter, and components, which will be covered in this entry and the two following it.

First up is counts. Counts literally count things in the graph. I'm looking at three generic measures:

- Node count
- Isolates count and percent
- Relationship count

These very basic metrics give you a feel for how big a graph is, how connected it is, and allows you to make a very general comparision with other graphs. However, keep in mind that these are very basic ways to describe a graph and can give a false impression that graphs are similar when they are in fact very different. This phenomenon of dissimilar things appearing similar was introduced way back in [Entry 5: Explore the Data](https://julielinx.github.io/blog/05_EDA/) when discussing descriptive statistics and Anscombe's quartet.

The notebooks where I did my code for this entry can be found on my github page. I created three notebooks, one for each graph model. These notebooks contain the code for Entries G6, G7, and G8.

- [Entries G6, G7, G8: Global Metrics Unimodal Graph Model](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8a_nb_unimodal_global_metrics.ipynb)
- [Entries G6, G7, G8: Global Metrics Biimodal Graph Model](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8b_nb_bimodal_global_metrics.ipynb)
- [Entries G6, G7, G8: Global Metrics Mixed Graph Model](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8c_nb_mixed_global_metrics.ipynb)
- A notebook with just the global graph counts can be found in the [Entry 14 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/14_nb_global_counts_comparison.ipynb)

*Note*, after I created a multigraph with all three graph models, the code changed significantly. You can read [Entry 14](https://julielinx.github.io/blog/g14_global_counts_comparison/) for the results of these changes, but that entry is a supplement to this one, not a replacement. 

## Node count

This is simply the number of nodes in the full graph.

In the unimodal model of the [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network) there are 6,439 nodes, which accounts for all the heroes.

The bimodal model of the graph has the 6,439 hero nodes plus the 12,651 comic nodes, which gives us a final node count of 19,090.

## Isolate count and percent

An isolated node is one that isn't connected to any other node.

For the Marvel Universe Social Network, it's possible to have isolated nodes in the unimodal model, but not the bimodal model. This is because a hero has to appear in a comic in order to be in the graph and a comic has to have a hero in order to have a story to tell.

There are 18 isolated nodes in the unimodal graph.

Because I was curious, I pulled up the 18 heroes that appeared alone in a comic. Keep in mind that the hero has to either be in only one comic where they appear alone or if they're in multiple comics, they have to be alone in those comics too (for a multiple appearance example, see Kull and Sea Leopard below). If a hero is alone in one comic, but appears in another one with other characters, that hero still won't end up on the isolate list because they would be connected to other heroes via the other comic(s).

<table align=left>
    <tr>
        <th>Hero</th>
        <th>Comic</th>
    </tr>
    <tr>
        <td>BERSERKER II</td>
        <td>FRANK 16</td>
    </tr>
    <tr>
        <td>BLARE/</td>
        <td>MTU2 1</td>
    </tr>
    <tr>
        <td>CALLAHAN, DANNY</td>
        <td>IM 12/34</td>
    </tr>
    <tr>
        <td>CLUMSY FOULUP</td>
        <td>SS3 53</td>
    </tr>
    <tr>
        <td>DEATHCHARGE</td>
        <td>NAMOR 4/2</td>
    </tr>
    <tr>
        <td>FENRIS</td>
        <td>JIM 114/2</td>
    </tr>
    <tr>
        <td>GERVASE, LADY ALYSSA</td>
        <td>IM:IA FB</td>
    </tr>
    <tr>
        <td>GIURESCU, RADU</td>
        <td>D:LD 1</td>
    </tr>
    <tr>
        <td>JOHNSON, LYNDON BAIN</td>
        <td>CA:SL 1</td>
    </tr>
    <tr>
        <td>KULL</td>
        <td>UX 13/3</td>
    </tr>
    <tr>
        <td>KULL</td>
        <td>SSOC 213/2</td>
    </tr>
    <tr>
        <td>LUNATIK II</td>
        <td>CPU 3/3</td>
    </tr>
    <tr>
        <td>MARVEL BOY II/MARTIN</td>
        <td>USA COMICS 7</td>
    </tr>
    <tr>
        <td>RANDAK</td>
        <td>AA2 10/2</td>
    </tr>
    <tr>
        <td>RED WOLF II</td>
        <td>M/SPT 1</td>
    </tr>
    <tr>
        <td>RUNE</td>
        <td>SS/EUNE/2</td>
    </tr>
    <tr>
        <td>SEA LEOPARD</td>
        <td>NAMOR 52</td>
    </tr>
    <tr>
        <td>SEA LEOPARD</td>
        <td>NAMOR 53</td>
    </tr>
    <tr>
        <td>SHARKSKIN</td>
        <td>WCA 5/3</td>
    </tr>
    <tr>
        <td>ZANTOR</td>
        <td>H' 98</td>
    </tr>
    </table>

## Relationship count

This is simply the number of relationships in the full graph.

The unimodal graph has 171,613 `KNOWS` relationships.

The bimodal graph has 96,104 `APPEARS_IN` relationships.

The mixed graph has both the 171,613 `KNOWS` relationships and the 96,104 `APPEARS_IN` relationships, so the final relationship count is 267,717.

## Count store

In the accompanying notebook I include several ways to count things. However, the count store can be a faster way to return this information.

One of the nice things about Neo4j is that you can access basic counts lightning fast with the count store. According to [Neo4j's Knowledge Base: Fast counts using the count store](https://neo4j.com/developer/kb/fast-counts-using-the-count-store/) "The count store is used to inform the query planner so it can make educated choices on how to plan the query." 

Often simple queries that use the `count()` function access the count store. This means that these queries are very fast.

If you want to be sure you're using the count store, you can put `EXPLAIN` in front of your query. If the count store is being used, you'll see `NodeCountFromCountStore` or `RelationshipCountFromCountStore`.

I haven't figured out if it's possible to get the results of an `EXPLAIN` query piped into a Jupyter notebook, but here's the screen capture of what it looks like when the query utilizes the count store:

`EXPLAIN MATCH (c)
RETURN count(c) as node_count`

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/plan_ct_store.png?raw=true'>

Now, let's not forget my paranoid side. I like to include `distict` in my queries to ensure I don't accidentally count the same data point more than once. This personality quirk allowed me to figure out that if you have a graph where a node can have multiple labels, the nodes with more than one label are double counted when queried certain ways and will end up in multi-label categories when queried another way.

An example of a multi-label node would be `MATCH (:Hero:CEO)` - which I expect would return Tony Stark if we had occupation data as labels. Depending on how the query is written, our theoretical Tony node could be counted all of the following ways:

- As a Hero node
- As a CEO node
- As a Hero|CEO node

*Note*, the count store isn't used for queries targeting multi-label nodes.

We'll get into this in more detail later when we have a more complex database with multi-labeled nodes. I believe the Movies dataset I plan to use next has nodes that are both actor and director. I'll be going through this with more than one dataset to streamline my processes and procedures while checking that they can be generalized to other datasets.

The point of this aside is to be careful how you write your queries.

Back to the point of using `distinct`, adding `distinct` to a query means the count store isn't used. Instead, it will scan the whole graph, which is much slower than using the count store, especially once we get to large graphs.

We can see this in the query plan for the following query:

`EXPLAIN MATCH (c)
RETURN count(distinct c) as node_count)`

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/plan_no_ct_store.png?raw=true'>

## Next Up

[Global Density and Diameter](https://julielinx.github.io/blog/g07_global_density_diameter/)

## Resources

- [Neo4j Python Driver 4.2](https://neo4j.com/docs/api/python-driver/current/)
- [Entry 5: Explore the Data](https://julielinx.github.io/blog/05_EDA/)
- [Fast counts using the count store](https://neo4j.com/developer/kb/fast-counts-using-the-count-store/)
- [Degree Centrality](https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality/)
- [Relationship Orientation](https://neo4j.com/docs/graph-data-science/current/management-ops/cypher-projection/#cypher-projection-relationship-orientation)
- [Entry 14: Global Counts Comparison](https://julielinx.github.io/blog/g14_global_counts_comparison/)