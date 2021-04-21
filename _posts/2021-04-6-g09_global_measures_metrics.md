---
title: "Entry G9: Measuring Performance"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
  - measuring performance
---

As I [may or may not have mentioned](https://julielinx.github.io/blog/g05_project_bimodal/), the last time I tried running many of the calculations that I'll be exploring in this series, I ran into timeout errors ... or crashed the cluster, which made me a persona non grata with the developer running the database.

Now that I'm trying different solutions to address these runtime problems, I need a way to compare performance of the different graph models, as well as assess the pros and cons of one over another.

The notebooks where I did my code for this entry can be found on my github page in the [Entry G9: raw notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/09a_nb_raw_global_meastures_metrics.ipynb) (i.e., the one with all my failed attempts to access the metadata) and [Entry G9: Global Measures notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/09b_nb_global_measure_metrics.ipynb) (i.e., the one with actual working code).

## The Problem

As discussed in [Entry G5](https://julielinx.github.io/blog/g05_project_bimodal/), I want to compare the memory use, processing speed, and index optimization for the different iterations of graph model I come up with.

Many blog posts point out that the retrieval time is listed at the bottom of the Neo4j Browser query results.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/runtime_highlighted.png?raw=true'>

This is great if I want to manually capture that information one query at a time. However, seeing that I have ~50 metrics and at least 3 graph models to run (not to mention iterating through different index optimizations), I really don't want to capture that information manually.

Additionally, as referenced in [Query Tuning](https://neo4j.com/docs/cypher-manual/current/query-tuning/) in the [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/), Neo4j caches query plans to help improve performance. This means that the first run of a query will be slower because the plan isn't cached. For this reason, as well as other causes of variation in run time, I want to run a query multiple times and capture the metrics for each run. This will allow me to see the differences in run times and memory use.

## The Options

I found two types of solutions to this problem:

- query plans
- performance monitoring

### Query Plans

`EXPLAIN` and `PROFILE` both return information about the query plan.

- `EXPLAIN`: returns the plan and expected results without actually running the query
- `PROFILE`: actually runs the query allowing it to return the query plan, actual run times, and the results of the query

The information for both `EXPLAIN` and `PROFILE` can be accessed using the `.consume()` method on the `Result` object.

### Monitoring

The other two options I found involve query logging or using the Halin Neo4j Monitoring app. 

Both of these options seem more suited to production monitoring of the performance/health of the database. Halin gives realtime results on the database, but tying that information to any one query would be tricky and involve manual monitoring. What I'm trying to determine is performance of the queries themselves.

The query logging option also appears to only be available in Enterprise Edition. That's not a problem for my work dataset, but I'm using the Community Edition to test my theories on toy datasets before moving back to the large work dataset. So the logging solution is out for these first tests.

## The Proposed Solution

After spending a significant amount of time with the Python Driver [Result](https://neo4j.com/docs/api/python-driver/current/api.html#result) documentation, the blog post [Neo4j Python 4.0 driver— the latest driver for the next gen database](https://medium.com/neo4j/neo4j-python-4-0-driver-the-latest-driver-for-the-next-gen-database-a5be6ecd481f), and finally giving up and talking to my software engineering friend Mohit, I was able to capture the `EXPLAIN` and `PROFILE` information for any given query.

The results of the `.consume()` method provides a wealth of information including:

- the time it took for the server to have the result available
- the time it took for the server to consume the result
- database hits
- global memory use
- number of rows returned
- time
- the same information as above but for each child step to achieve the final result

## The Fail

I haven't needed this header for a while. But...

When pulling back the metadata on the query Result objects, I found that some queries have more information and some have less. Of the data points I'm interested in, not all of them are available for all queries. Specifically, queries like `call apoc.meta.stats()` were missing some data points I was trying to pull.

The complexity of a query also determines how many child steps it has. A query like `MATCH (c) RETURN count(c) as node_count` has one child entry. 

This query has nine child levels:

`MATCH (n) WHERE NOT (n)--() 
WITH COUNT(distinct n) as isolates_count
MATCH ()-[r]->()
WITH count(r) as relation_ct, isolates_count
MATCH (c)
with count(distinct c) as node_count, isolates_count, relation_ct
return node_count, relation_ct, isolates_count, round(toFloat(isolates_count)/node_count*10000) / 100 as isolates_pct`

If I want information on child steps, I'll need to consume this information dynamically.

At this point, I'm not sure I even need that information. If I were working on optimizing a query (which I may very well need to do at some point), I'd want the child level data so I could optimize the steps and hone the results. However, that feels like a separate task from simply measuring how the queries perform across different graph models.

### Decision

For now, I'll keep in mind that the information I want is accessed through `.consume()`. I'll collect that data for all the queries after I've determined how I want to capture them (i.e., once I've gone through the full list of analysis metrics), then narrow down the fields available for all query Result objects or determine whether I need to tailor the fields for each specific query.

That will just leave capturing the metadata for multiple runs of the query; calculating the minimum, maximum, and standard deviation of those runs; and figuring out the best way to start and stop the different graph models (or store them as different databases in the same instance, which is now possible thanks to version 4).

## Up Next

[Local Metrics](https://julielinx.github.io/blog/g010_local_metrics/)

## Resources

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Query Tuning section of the Cypher Manual](https://neo4j.com/docs/cypher-manual/current/query-tuning/) 
- [Python Driver API Documentation](https://neo4j.com/docs/api/python-driver/current/api.html)
- [Result section of the API Documentation](https://neo4j.com/docs/api/python-driver/current/api.html#result)
- [Neo4j Python 4.0 driver— the latest driver for the next gen database](https://medium.com/neo4j/neo4j-python-4-0-driver-the-latest-driver-for-the-next-gen-database-a5be6ecd481f)
