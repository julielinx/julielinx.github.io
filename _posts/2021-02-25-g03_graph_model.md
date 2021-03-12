---
title: "Entry G3: Choosing a Graph Model (schema)"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph model
---

To harness the power of graph, data first has to be organized to fit the node/relationship format.

## Graph Model Types

Just like with a relational database, when you decide to put information in a graph database you have to have some kind of schema. In the graph world this is called *graph modeling* or *the graph model*. There are three general graph models:

- Unimodal
- Bimodal
- Multimodal

#### Caution

These graph models have other other names as well, most notably k-partite. I will use the unimodal, bimodal, and multimodal terms consistently through this series of  posts, but keep in mind that in graph literature, there are other ways to refer to these same models.

In addition, unimodal, bimodal, and multimodal are also used to describe distributions of data. This becomes confusing when "graph" is used instead of "chart" to describe the visual representation of these distributions.
 
As such, a chart of a bimodal distribution would be called a "bimodal graph." These uses have very different contexts and refer to very different things. For reference, here is a chart of a bimodal distribution:

<img src='https://upload.wikimedia.org/wikipedia/commons/e/e2/Bimodal.png'>


### Unimodal

In a unimodal graph, there is only one kind of node. For the first database I'll be working with, the [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network), this means we'd only have hero nodes. We wouldn't include the comics that the heroes were featured in.

#### Simplified View

Here's a simplified view of what a unimodal graph would look like:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/unimodal_heroes.png?raw=true'>

#### More Realistic View

Of course, data isn't that simple and easy to read. For a more realistic view of what this would look like, I created a small subset of the Marvel Universe Social Network centering around heroes and comics connected to Dark Crawler. I pulled these into Gephi and after a little finagling got this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_unimodal.png?raw=true'>

There is a reason some of the lines are thicker than others. We'll cover that in the next entry, [Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/).

#### Benefits

There are a couple of benefits to a unimodal setup like this:

1. Queries from one hero to another are easier
  - Instead of `(hero1)-[:APPEARS_IN]-(comic)-[:APPEARS_IN)-(hero2)` we can just write `(hero1)-[:KNOWS]-(hero2)`
2. We can put a weight on the relationship, which will tell us how many comics the heroes are in together
3. To reduce data loss, we can also create a list on the relationship that will tell us the names of the comics they appear in together

### Bimodal

In a bimodal graph, there are exactly two kinds of nodes. These nodes usually don't connect to their own node type, only the other node type.

The Marvel Universe Social Network example naturally lends itself to a bimodal structure because there are 'hero' nodes and 'comic' nodes. Heroes are in comics and we can tell what heroes appear together by connecting them through the comics where they both appear. Heroes aren't connected to other heroes, they're only connected to the comics they appear in. Comics also aren't connected to each other, only to the heroes that appear in their pages.

#### Simplified View

The image below shows two ways to visualize the same bimodal graph.

- The left hand version clearly shows that Hero nodes only connect to Comic nodes and vice versa.
- The right hand version more clearly shows which nodes are connected to each other and how they relate as a group.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/bimodal.png?raw=true'>

#### More Realistic View

Using the same subset of data from the unimodal example above (Dark Crawler and his compatriots now also including the comic information), we get a graph that looks like this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_bimodal.png?raw=true'>

Here are some of the benefits of this structure:

1. It is intuitive for many use cases
2. It appears visually less cluttered because there are fewer relationship lines
3. There is no data loss from cutting out the "middle man" comic nodes

### Multimodal

In a multimodal graph, there are more than two kinds of nodes. Like with bimodal models, in the multimodal model nodes generally don't connect directly to the same kind of node.

#### Simplified View

Retail data naturally fits the multimodal graph model. We have things like 'customers', 'orders', and 'shipping addresses'. In the simplified visualization below, we can think of the nodes as follows:

- Green: customers
- Teal: orders
- Pink: credit card tokens
- Orange: shipping addresses
- Brown: phone numbers
- Blue: IP address

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/multimodal.png?raw=true'>

#### More Realistic View

The Mavel Universe Social Network doesn't lend itself to the multimodal model because there are only two kinds of nodes. But for consistency sake, I wanted to use it as the example. To do this we have to add more information, like planets the characters have visited. In this case we now have 'hero' nodes, 'comic' nodes, and 'planet' nodes.

*Note*, don't look too close at the planets the heroes have visited. I'm not that deep in the Marvel universe to know off-the-cuff what planets each of the characters have been to and only wanted to spend so much time (not much) looking up such things. Mostly, I guessed.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_multimodal.png?raw=true'>

#### Benefits

The main benefit of the multimodal structure is preservation of data.

### Projected Unimodal

Some data, like the Marvel Universe Social Network, naturally lends itself to a bimodal or multimodal graph model. When this is the case, but it's represented as a unimodal graph instead, it's called a projected unimodal graph. This is because the more complex graph model is projected onto the more simplified unimodal representation.

When this is done there is inevitably data loss. However, the loss of information is not always a bad thing. For some use cases the data that is lost is extraneous and its absence goes unnoticed.

I'll go into more detail on projected unimodal graphs in [Entry G5](https://julielinx.github.io/blog/g05_project_bimodal/).

## Up Next

[Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/)

## Resources

For a more interactive introduction to graphs, graph visualization, and graph modeling, watch my [Not All Visualizations are Created Equal](https://neo4j.com/videos/24-not-all-visualizations-are-created-equal/) talk at NODES2020.

- [Networks](https://www.amazon.com/Networks-Mark-Newman/dp/0198805098) by Mark Newman
- [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network)
- [Gephi](https://gephi.org/)
- [Not All Visualizations are Created Equal](https://neo4j.com/videos/24-not-all-visualizations-are-created-equal/)
- [Entry G4: Modeling Relationships](https://julielinx.github.io/blog/g04_graph_model_rels/)
- [Entry G5: Projecting Bimodal to Unimodal](https://julielinx.github.io/blog/g05_project_bimodal/)