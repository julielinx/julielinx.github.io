---
title: "Entry G10: Local Metrics"
categories:
  - Blog 
tags:
  - graph
  - neo4j
  - graph analytics
---

Now that I know what the larger graph looks like, I need metrics at the node level. The reason for these metrics is to be able to locate outlier nodes within the graph for closer inspection. Pretty much all queries I'll run from here out are to identify nodes for additional inspection or exclude them from consideration.

There are six metrics I'll be running for this entry:

1. Degree count
2. Degree summary statistics
3. Degree distribution
4. Weighted degree count
5. Weighted degree summary statistics
6. Weighted degree distribution

For my purposes, summary statistics include min, max, mean, median, and standard deviation

The notebooks where I did my code for this entry can be found on my github page.

I created three notebooks, one for each graph model.

- [Entry G10a: Unimodal Model Local Metrics](https://github.com/julielinx/datascience_diaries/blob/master/graph/10a_nb_uni_local_metrics.ipynb)
- [Entry G10b: Bimodal Model Local Metrics](https://github.com/julielinx/datascience_diaries/blob/master/graph/10b_nb_bi_local_metrics.ipynb)
- [Entry G10c: Mixed Model Local Metrics](https://github.com/julielinx/datascience_diaries/blob/master/graph/10c_nb_mix_local_metrics.ipynb)

## Degree Count

The first query I run in the notebooks is the same as the relationship count that I ran in the [Entry G6a notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/06_7_8a_nb_unimodal_global_metrics.ipynb) (this is for the unimodal model, but the relationship query is pretty much the same regardless of which model we're using). While it is technically a global metric, I'm including it here as a reminder of how many relationships we have in the graph.

Next we get the degree count for each and every node in the database.

![Degree Counts by Node](https://julielinx.github.io/assets/images/g10_degree_df.png)

This is a lot of data to parse, even if we only use the 6,439 Hero nodes. I am both too lazy and too short on time to look at the degree count for every single node. Also, there's no guarantee a manual review of every value would reveal anything interesting anyway. The answer? Descriptive Statistics and Distribution Charts.

## Degree Summary Statistics

Descriptive statistics are a way to quantitatively describe or summarize a collection of information. I know because [Wikipedia](https://en.wikipedia.org/wiki/Descriptive_statistics) told me so. Okay, fine. Wikipedia was just the easiest (and most reliable to be around next month) source.

I was just going to leave it at "descriptive statistics" and call it a day, but Wikipedia said no, I needed to be more specific. So, within descriptive statistics are **summary statistics**. The [Wikipedia definition](https://en.wikipedia.org/wiki/Summary_statistics) is very nice: "summary statistics are used to summarize a set of observations, in order to communicate the largest amount of information as simply as possible."

I found four ways to get the summary statistics for the local degree counts:

- Pattern match
- `apoc.node.degree` function
- `gds.alpha.degree.stream` function
- `size` function (but I didn't find this until working on Entry 12, so the notebooks accompanying Entry 12 are where you'll find the code using this function)

### Pattern Match

Writing a direct Cypher query is the most straight forward way to grab this information. 

Initially I used the pattern `(c1)-[]-(c2)`, which only returns nodes that have some connection (i.e., the `-[]-()` portion of the query forces the node to have a relationship of some kind). This means that isolated nodes won't be returned and the minimum will always be greater than 0.

After I'd completed all 3 notebooks for this post it occurred to me I could also use a pattern match with an "optional" component instead (`MATCH (c1) OPTIONAL MATCH (c1)-[]-(c2)`). This is a slightly more advanced query, but much more useful in comparing the exact same results between methods. I switch to the "optional" method in Entry 12.

### Provided Functions

The `apoc` and `gds` functions both include isolated nodes in their counts. As such, the minimum will be 0 whenever there are isolated nodes in the graph.

### `apoc` vs `gds` Comparison

The two functions return the same results for simple queries, but the `gds` option appears more limited when we get into bimodal type models. I go into this in more detail in Entry 12.

The average and standard deviation of the functions vs the pattern match is slightly different because the 18 isolate nodes are removed from the calculations for the direct query due to the limitations of the pattern match.

## Degree Distribution

From the definition of summary statistics above, I especially like the part where it says they're used "to communicate the *largest* amount of information as *simply* as possible." Now, I know I just referenced Anscombe's Quartet back in [Entry G6](https://julielinx.github.io/blog/g06_global_counts/) (citing the original reference from [Entry 5](https://julielinx.github.io/blog/05_EDA/)), but it bears repeating. Summary statistics give a good amount of information about a dataset, but due to the simplification they can be misleading if they aren't seen in context.

Distribution charts help give us the context to appropriately interpret the summary statistics.

All three graph models (unimodal, bimodal, and mixed) clearly show an exponential decay in the number of nodes for higher degrees. Albert-László Barabási states in [Network Science](http://networksciencebook.com/chapter/2#real-networks) that most real world networks are sparse networks, i.e. real world networks usually have a low density (density is covered in [Entry G7](https://julielinx.github.io/blog/g07_global_density_diameter/)).

For the density to be low, the number of relationships must be significantly lower than the number of possible relationships. For this to be true, there must be more nodes that have fewer connections.

Consider a social network like LinkedIn as an example. Most people have tens to a few hundred connections. The number of people with thousands or millions of connections are very small. Same for social media networks like Twitter. Only celebrities have very high numbers of relationships.

When we look at this kind of data on a normal scale, the values at the high end tend to be very hard to see. I made the y-scale logarithmic to make the high end easier to interpret. A clear, linear decay becomes evident up to a certain point, at which the number of nodes jumps erratically before permanently decreasing to 0.

## Weighted Version

Weighted degree is just the degree with a weight to take into account the strength of the connection.

When considering the original bimodal model, there is no direct weight. Either a hero is in a comic or they aren't. To get a weight, we'd need to consider projected Hero to Hero connections or projected Comic to Comic connections (I discussed projecting bimodal models to unimodal models in [Entry G5](https://julielinx.github.io/blog/g05_project_bimodal/)). Point of fact, this is what we did to create the unimodal version of the [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network).

In the unimodal model, the weight has already been calculated and stored for us. All we have to do is pull up the weights and add them together.

I skipped weighted degrees for the bimodal model since the relationships have no weight, nor would we necessarily want one when looking at Hero to Comic relationships.

In the mixed model, I projected Hero to Hero and Comic to Comic weights, mostly just to prove that I could. The numbers for the Hero to Hero projection matched the numbers in the Unimodal Model. This makes me more confident that everything worked the way it was supposed to during the projection process.

## The Fail

You've probably noticed by now the fact that I'm doing three separate notebooks for each topic, one for each model type; unimodal, bimodal, and mixed. It's really annoying to have to shut down the graph and fire up a different one.

It's also really difficult to compare stuff across models when I have to manually start and stop databases in another application. This makes the notebooks messy and harder to explain since I'm always having to specify which database to activate. And don't even get me started on how annoying it is to switch from notebook to notebook when looking at a bunch of numbers.

It's time to figure out the multigraph capabilities of Neo4j version 4.

## Next Up

[Create a Multigraph Database](https://julielinx.github.io/blog/g011_create_multigraphdb_desktop/)

## Resources

- [Neo4j Python Driver 4.2](https://neo4j.com/docs/api/python-driver/current/)
- [Fast counts using the count store](https://neo4j.com/developer/kb/fast-counts-using-the-count-store/)
- [Degree Centrality](https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality/)
- [Relationship Orientation](https://neo4j.com/docs/graph-data-science/current/management-ops/cypher-projection/#cypher-projection-relationship-orientation)
- [How to get Degree centrality in bipartite graph by using GDS?](https://community.neo4j.com/t/how-to-get-degree-centrality-in-bipartite-graph-by-using-gds/30278)
