---
title: "Entry G16: Components Comparison"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

The [notebook that accompanies this entry](https://github.com/julielinx/datascience_diaries/blob/master/graph/16_nb_components_comparison.ipynb) is a cleaned up, concise version of the three notebooks I created for [Entry G8](https://julielinx.github.io/blog/g08_components/), addressing just the graph components. Just like [Entry G14](https://julielinx.github.io/blog/g014_global_counts_comparison/), this is a supplement to, not a replacement for, the older entry ([G8](https://julielinx.github.io/blog/g08_components/)) so make sure you read the older one first.

This was the easiest of the redo entries/notebooks. There were two surprises when I put everything together. The first struck me as really strange: the Bimodal and Mixed Models have the exact same components; ids, sizes, everything.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/marvel_components.png?raw=true'>

At this point, I can only guess that the `KNOWS` relationships play such a small part in the shape of the Mixed Model graph that they're negligible. This would leave the algorithm to find the same components in both the Bimodal and Mixed Models.

I ran the algorithm a couple of times with the same results. While weird, it is nice knowing that the results are reproducible.

## Comparison

The three graph models have very similar components. All three had exactly 22 components (which was the second surprise; I expected them to have different numbers of components). I'm interested to see if the number of components stays similar when we get to other datasets or whether the number of components stays the same between graph models regardless of whether it's a unimodal, bimodal, or mixed representation.

I have run the components algorithm on other datasets with a very different number of components, so I do know that it returns a different number than only 22.

What I did expect was a giant component that consisted of the majority of the graph: all three graph models had a giant component comprising over 99% of the nodes. This means that less than 1% of nodes in all three models aren't connected to the rest of the nodes in some way, shape, or form.

This aligns with the findings of Mark Newman in [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098), Second Edition on page 305, as previously referenced in [Entry G8](https://julielinx.github.io/blog/g08_components/). Of the 27 different graphs from 4 different industries only 3 of them had a giant component consisting of less than 80% of the nodes. I expect the other example graphs I'll be running will show the same giant component.

# Up Next

[Add Villains](https://julielinx.github.io/blog/g17_add_villains/)

# Resources

- [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098) by Mark Newman
- [Entry G8](https://julielinx.github.io/blog/g08_components/)