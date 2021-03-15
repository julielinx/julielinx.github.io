---
title: "Entry G5: Projecting Bimodal to Unimodal"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - create database
---

I need to understand graph structure better and the repercussions of using the different model types. Specifically, I'm interested in memory use, processing speed, and index optimization for the ~50 different graph metrics and algorithms I've explored for machine learning features.

As mentioned in [Entry G2](https://julielinx.github.io/blog/g02_create_graphdb_desktop/) I'm using the [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network) as a jumping in point. This data is handy because:

- It's public
- It's easy to load into a Neo4j graph database
- It easily fits either a bimodal or unimodal structure
- It's small enough that I can store multiple versions of it on a laptop

## Database Versioning

The bimodal graph structure is the one that I work with for my job (technically it's multimodal, but I tend to think of it as bimodal). During the course of my work, I've started to wonder several things:

- Is it necessary to project a bimodal graph to a unimodal graph to run the graph algorithms?
- Is it easier to use a projected unimodal graph when looking to engineer features for machine learning?
- Is it faster to use a bimodal or unimodal structure (historically I have trouble with timeout errors)?
- Can a unimodal and bimodal version of the graph exist in the same space and still be usable?

In order to really explore the ramifications of the different graph models (see [Entry G3](https://julielinx.github.io/blog/g03_graph_model/) for more on graph modeling), I created three versions of the Marvel Universe Social Network. The three versions are:  

1. Bimodal
2. Weighted projected unimodal
3. Mixed bimodal and projected unimodal

### Context

Why create three different versions? Why not just use the mixed graph and have it all? There are a couple reasons.

First, I frequently get a mismatch between what I expect and the actual results. As such, I tend to test things from multiple angles before trusting my results.

During these tests, I've found that not all functions do what I think they will. For example, I ran the `gds.alpha.degree.stream()` function from the Graph Data Science Library. The [Degree Centrality](https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality/) doc page states "Degree centrality measures the number of incoming and outgoing relationships from a node."

I used the `count()` function (which allowed me to easily hone in on different relationship populations: hero-to-comic or hero-to-hero or the combination of both - correct results below) to double check the results and found that the number of relationships wasn't accurately reflected in the numbers the function returned. 

*Side note*, if you don't know Cypher and don't understand the queries or syntax, don't worry too much. I'll go into more detail once we get into the metrics and running actual queries. If you want or need to know right now, check out the [Introduction](https://neo4j.com/docs/cypher-manual/current/introduction/#cypher-intro), [Syntax](https://neo4j.com/docs/cypher-manual/current/syntax/#query-syntax), [Clauses](https://neo4j.com/docs/cypher-manual/current/clauses/#query-clause), and [Functions](https://neo4j.com/docs/cypher-manual/current/functions/#query-function) sections of the [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/).

#### Hero-to-hero degree count

```
MATCH (h:Hero)-[r]-(o:Hero)
RETURN h.name, count(r) as h_degree
Order by h_degree desc
```

Top 5 results:

<table>
    <tr>
        <th>Hero</th>
        <th>Degree Count</th>
    </tr>
    <tr>
        <td>"CAPTAIN AMERICA"</td>
        <td>1919</td>
    </tr>
    <tr>
        <td>"SPIDER-MAN/PETER PARKER"</td>
        <td>1754</td>
    </tr>
    <tr>
        <td>"IRON MAN/TONY STARK"</td>
        <td>1566</td>
    </tr>
    <tr>
        <td>"THING/BENJAMIN J. GR"</td>
        <td>1448</td>
    </tr>
    <tr>
        <td>"MR. FANTASTIC/REED R"</td>
        <td>1416</td>
    </tr>
</table>

#### Hero-to-comic degree count

```
MATCH (h:Hero)-[r]-(o:Comic)
RETURN h.name, count(r) as h_degree
Order by h_degree desc
```

Top 5 results:

<table>
    <tr>
        <th>Hero</th>
        <th>Degree Count</th>
    </tr>
    <tr>
        <td>"SPIDER-MAN/PETER PARKER"</td>
        <td>1577</td>
    </tr>
    <tr>
        <td>"CAPTAIN AMERICA"	</td>
        <td>1334</td>
    </tr>
    <tr>
        <td>"IRON MAN/TONY STARK"</td>
        <td>1150</td>
    </tr>
    <tr>
        <td>"THING/BENJAMIN J. GR"</td>
        <td>963</td>
    </tr>
    <tr>
        <td>"THOR/DR. DONALD BLAK"</td>
        <td>956</td>
    </tr>
</table>

#### Hero-to-all degree count

```
MATCH (h:Hero)-[r]-(o)
RETURN h.name, count(r) as h_degree
Order by h_degree desc
```

Top 5 results:

<table>
    <tr>
        <th>Hero</th>
        <th>Degree Count</th>
    </tr>
    <tr>
        <td>"SPIDER-MAN/PETER PARKER"</td>
        <td>3331</td>
    </tr>
    <tr>
        <td>"CAPTAIN AMERICA"</td>
        <td>3253</td>
    </tr>
    <tr>
        <td>"IRON MAN/TONY STARK"</td>
        <td>2716</td>
    </tr>
    <tr>
        <td>"THING/BENJAMIN J. GR"</td>
        <td>2411</td>
    </tr>
    <tr>
        <td>"HUMAN TORCH/JOHNNY S"</td>
        <td>2298</td>
    </tr>
</table>

Second, I want multiple versions of the graph to see if it's easier and faster to structure it one way vs another. One of the major road blocks I've encountered at work is that it takes forever to run some of the queries and algorithms I want. These problem queries end up timing out or temporarily crashing the graph. As would be expected, our developers and software engineers get rather testy when that happens.

# Unimodal Projection

## Context

The purpose of projecting a bimodal graph to a unimodal structure is to directly connect nodes of interest.

As a reminder from [Entry G3](https://julielinx.github.io/blog/g03_graph_model/), here's what it looks like when we take a bimodal graph and project it to a unimodal graph:

#### Bimodal Graph

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_bimodal.png?raw=true'>

#### Projected Unimodal Graph

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_unimodal.png?raw=true'>

### Cliques Caveat

One thing to keep in mind when projecting a bimodal network into a unimodal structure is that you'll get a lot of *cliques*.

**Clique**: a subset of nodes where every distinct node is connected to every other distinct node.

There are four cliques from our example above.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/model_unimodal_cliques.png?raw=true'>

To make this concept clearer, let's look at an example from a single comic.

I pulled the comic W2 50 and the heroes in it. Some statistics about this subgraph:

- There is 1 comic
- There are 9 heroes
- Total of 10 nodes
- There are 9 relationships

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/clique1.png?raw=true'>

When we project this to a unimodal structure, the number of relationships multiples significantly. The statistics for the projected version:

- There are 0 comics (we removed it so we could connect heroes directly)
- There are 9 heroes
- Total of 9 nodes
- There are 36 relationships

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/clique2.png?raw=true'>

Another caveat to this method is that projecting the relationships makes certain assumptions about the connectivity of the other nodes. For example, in the W2 50 comic example, Silver Fox may not have actually met some of the other characters. [Wikipedia](https://en.wikipedia.org/wiki/Silver_Fox_(comics)) says she is the former love interest for Wolverine. The only X-man she interacted with could have been Wolverine, but in projecting the bimodal graph to a unimodal graph we are making the assumption that all nodes (heroes) interacted with each other because they were in the same comics.

I'm not quite sure what the repercussions of these caveats are for the metrics I want and the populations I'm trying to find, but that's one of the questions I'm trying to answer in this series of entries.

## Create the Mixed Model

The raw data for the Marvel Universe Social Network is essentially stored as a bimodal graph, so that was my starting point. If you remember, we loaded the Marvel data into a bimodal graph in [Entry G2](https://julielinx.github.io/blog/g02_create_graphdb_desktop/).

Now, you could go through all the steps in [Entry G2](https://julielinx.github.io/blog/g02_create_graphdb_desktop/) two more times to create the base graph, or you could simply clone what you already did.

#### 1. Clone the bimodal graph

Click on the dots in the upper right corner of the database on the My Project page and choose `Clone`.

**Caution**: When you click `Clone` it looks like nothing is happening. Give it a minute and a new database named *DBMS* should appear in your list of databases.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/clone_db1.png?raw=true'>

#### 2. Rename the database

The default database name is `DBMS`. To change it simply select the database, which will bring up the `Details` panel. Click the pencil icon near where it says `DBMS` and type whatever name you want (unsurprisingly, I named mine "Marvel Universe Mixed").

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/rename_db.png?raw=true'>

#### 3. Add weighted edges to the graph

Start the newly cloned graph and enter the following code into the Neo4j command line:

```
Call apoc.periodic.iterate('MATCH (h1:Hero)-->(:Comic)<--(h2:Hero) where id(h1) < id(h2) RETURN h1, h2',
'MERGE (h1)-[r:KNOWS]-(h2) on CREATE SET r.weight = 1 on MATCH SET r.weight = r.weight+1', {batchSize:5000, parallel:false, iterateList:True});
```

#### 4. Admire your new mixed model graph

If you want to take it for a test spin, go to the section below titled "Code for Bimodal/Unimodal Examples" (below the "Resources" section) and try out the code to find the subsets I used for the examples.

## Create the Projected Unimodal Model

#### 1. Clone the mixed graph

Since the mixed model graph has both bimodal and unimodal nodes and relationships, we can just start from there.

Just like before, click on the dots in the upper right corner of the appropriate database (this time the mixed graph instead of the bimodal graph) on the My Project page and choose `Clone`.

#### 2. Rename the database

The default database name will still be `DBMS`. To change it select the database, which will bring up the `Details` panel. Click the pencil icon near where it says `DBMS` and type whatever name you want (unsurprisingly, I named mine "Marvel Universe Unimodal").

#### 3. Remove bimodal relationships

To create a truly unimodal graph, we need to remove the comics and all the relationships that connect to the comic nodes.

Start the newly cloned graph (make sure your other graphs are stopped or the new one won't start). Then enter the following code into the Neo4j command line:

```
MATCH (c:Comic)
DETACH DELETE c;
```

The `DETACH DELETE` command conveniently deletes the selected nodes (in this query, everything with the `Comic` label) and all the relationships connected to those nodes.

#### 4. Admire your new unimodal model graph

## Next Up

[Global Graph Counts](https://julielinx.github.io/blog/g06_global_counts/)

## Resources

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Graph Data Science Manual: Degree Centrality](https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality/)
- [Neo4j Cypher Manual: Delete](https://neo4j.com/docs/cypher-manual/current/clauses/delete/)
- [Graph People: Neo4j to Gephi](https://tbgraph.wordpress.com/2017/04/01/neo4j-to-gephi/)
- [Graph People: Neo4j Marvel Social Graph](https://tbgraph.wordpress.com/2017/06/10/neo4j-marvel-social-graph/)
- [Entry G2: Create a Neo4j Database](https://julielinx.github.io/blog/g02_create_graphdb_desktop/)
- [Entry G3: Choosing a Graph Model](https://julielinx.github.io/blog/g03_graph_model/)

## Code for Bimodal/Unimodal Examples

If you want to play with this data yourself, here are the steps I followed to get the subsets and send them to Gephi.

*Note*, when locating the subsets I used the mixed model which has both the hero-to-hero connections AND the hero-to-comic connections.

#### Dark Crawler

1. Find a small subset

To find a small subgraph for the first example, I did a degree count (in this case the degree count will return the number of comics that the hero appears in according to our dataset), then picked a name from the list.

The query has the condition that the hero must be in more than 4 comics. From the results, I picked "DARK CRAWLER" because it was close to the top and it looked vaguely familiar and was easy to spell (I wasn't really interested in typing "YASHIDA, MARIKO | MU" every time I wanted to run a query).

```
MATCH (h:Hero)-[r]-(o:Comic)
WITH h.name as h_name, count(r) as h_degree
WHERE h_degree > 4
RETURN h_name, h_degree
Order by h_degree
```

2. Send Hero Dark Crawler’s hero connections to Gephi

```
MATCH path = (h1:Hero {name: 'DARK CRAWLER'})-[:KNOWS]-(h2)
CALL apoc.gephi.add(null, 'workspace1',path,'weight') yield nodes
RETURN *
```

3. Send Hero Dark Crawler’s comic-hero connections to Gephi

```
MATCH path = (h1:Hero {name: 'DARK CRAWLER'})-[:APPEARS_IN]-(c:Comic)-[:APPEARS_IN]-(h2:Hero)
CALL apoc.gephi.add(null, 'workspace2', path) yield nodes
RETURN *
```

#### W2 50

1. Find a small subset

For the second example, I wanted a comic with a small number of connections but enough to make the cliques obvious, so I counted all relationships of the heroes, then scrolled to the end of the list. "SHIVA" had nine connections, which seemed like it would fit the bill (yes, there was some trial and error in choosing a hero to use).

```
MATCH (h:Hero)-[r]-(o)
RETURN h.name, count(r) as h_degree
Order by h_degree
```

2. Send Hero Shiva’s hero connections to Gephi

```
MATCH path = (h1:Hero {name: 'SHIVA'})-[:KNOWS]-(h2)
CALL apoc.gephi.add(null, 'workspace1',path,'weight') yield nodes
RETURN *
```

3. Send Comic connections to Gephi

This query results in the same subset as step 2, but starts from the comic name instead of the hero.

```
MATCH path = (c:Comic {name: 'W2 50'})-[:APPEARS_IN]-(h:Hero)
CALL apoc.gephi.add(null, 'workspace2', path) yield nodes
RETURN *
```