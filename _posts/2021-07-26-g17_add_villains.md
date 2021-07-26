---
title: "Entry G17: Add Villains"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

The use case I'm interested in is locating fraud within a graph. This is difficult to do when all the people nodes in your data are undifferentiated.

In perusing the names in our [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network), it occurred to me that there are villains in the dataset despite the fact that they all have the label "Hero." Now, we could debate whether characters like Deadpool, who are sometimes heroes and sometimes villains, are heroes, but for the sake of argument, if I could track a character back as some type of villain, they got a "Villain" label.

I spent a day or two comparing characters on [Villains Wiki](https://villains.fandom.com/wiki/Category:Marvel_Villains) to the characters in the Marvel Universe Social Network and making a list of everything that overlapped. I was able to automate a large portion of that comparison, but some of the entries needed manual review and use of "CONTAINS" instead of exact matches.

The accompanying [Entry 17 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/17_nb_add_villains.ipynb) has the results of that work. All you need to do is verify the connection, login, and database name information and run the notebook. It will automatically update the labels to include "Villain" in all 3 of the graphs of our [Marvel Multigraph](https://julielinx.github.io/blog/g11_create_multigraphdb_desktop/).

If you're still using the separate graphs and haven't moved to the multigraph solution, go back to [Entry 11](https://julielinx.github.io/blog/g11_create_multigraphdb_desktop/) (and more importantly the [Entry 11 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/11_nb_create_multigraphdb.ipynb)) and create the multigraph. All entries will be using the convenience afforded by the multigraph from here on out.

I've had some computer problems the last couple of weeks and have used the [Entry 11 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/11_nb_create_multigraphdb.ipynb) to recreate the multigraph at least thrice (once on my temporary work laptop replacement, once on my reformatted permanent work laptop, once on my new personal laptop) and it's really simple - I promise.

## Up Next

Egocentric Network

# Resources

- [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network)
- [Villians Wiki](https://villains.fandom.com/wiki/Category:Marvel_Villains)
- [Entry 11: Create the Marvel Multigraph](https://julielinx.github.io/blog/g11_create_multigraphdb_desktop/)
- [Entry 11 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/11_nb_create_multigraphdb.ipynb)
- [Entry 17 notebook](https://github.com/julielinx/datascience_diaries/blob/master/graph/17_nb_add_villains.ipynb)
