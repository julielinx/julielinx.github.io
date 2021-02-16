---
title: "Entry G2: Create a Neo4j Database"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - create database
---

To begin any data science project, I need a data to play with. For the graph project I'm going to start with the [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network) available on Kaggle. I picked this because it's publicly available, stored as .csv files, and easily fits into both bimodal and unimodal graph models.

### Get Database software

I work with Neoj4 databases at work, so that's what I'll use here too. This decision was based on my familiarity with the product as well as the fact that Neo4j has an open source [Community Edition](https://neo4j.com/download-center/?ref=web-product-database/#community) and free [Desktop edition](https://neo4j.com/download-center/?ref=web-product-database/#desktop).

I'll be using the Desktop edition for the examples. To follow along, download the version of the free Neo4j [Desktop edition](https://neo4j.com/download-center/?ref=web-product-database/#desktop) appropriate for your computer and follow the directions.

When you've got it loaded and fired up, you should have a page like this, but with an empty My Projects area:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/neo4j_desktop.png?raw=true'>

### Create an Empty Database

Now we need a new, empty graph that we can load the Marvel data into. This is easy and straight forward:

#### 1. Make sure you're on database page. This is the database stack icon in the far left.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/dbstack_icon.png?raw=true'>

#### 2. Click the `Add` button in the upper right of the My Project area

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/dbadd_button1.png?raw=true'>

#### 3. Choose `Local DBMS`

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/dbadd_button2.png?raw=true'>

#### 4. Give your database a name and set a password and version

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/db_create.png?raw=true'>

#### 5. Admire your new database

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/create_directions.png?raw=true'>

### Setup the database

Next we need to prepare the database for data.


#### 1. Click on your new database

This opens the Options for the database. It defaults to the Details page.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/options_default.png?raw=true'>

#### 2. Navigate to `Plugins`

We want to add some plugins to make working with the data easier. To do this, choose the `Plugins` tab. You'll see the four available plugins

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/options_plugins.png?raw=true'>

#### 3. Install the desired plugins
  
I generally use the APOC library and Graph Data Science Library
  - APOC: this library holds a lot of useful, optimized functions that make writing queries easier
  - Graph Data Science Library: this library holds a lot of useful, optimized functions that are specifically designed for data science and analytic purposes
    
1. Expand the library you want
2. Click the `Install` button

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/options_installplugins.png?raw=true'>

*Side note* You can add/remove/change the options at any time (now or after loading the data). However, if the database is running it takes longer to add plugins because the database has to restart for every library you install.

### Import data

Now that our database is ready, we can import data.

Tomaz Bratanic, who often posts about graph and Neo4j on his [Graph People blog](https://tbgraph.wordpress.com/), kindly hosts the network on [his github page](https://github.com/tomasonjo), which is easier to connect to than the Kaagle page, so we'll import the data from there.

#### 1. Start the database

Click `Start` to start the database

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data1.png?raw=true'>

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data2.png?raw=true'>

#### 2. Open the database

Click `Open` to fire up the browser based interface

You'll notice that once the database is running, the options change slightly; `Start` has changed to `Stop` and `Open` is now selectable.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data3.png?raw=true'>

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data4.png?raw=true'>

#### 3. Navigate to Neo4j Browser

You should now have a Neo4j Desktop Browser window open. The initial page usually looks something like this (I'm using Neo4j Desktop version 1.4.1 and database version 4.1.3)

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data5.png?raw=true'>

#### 4. Set the schema

Since we're using the Marvel Universe Social Network, we want two node labels: "Hero" and "Comic". To do this, we'll use the Neoj4 Broswer command line.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/neo_cmdline.png?raw=true'>

Just input the following Cypher code
  
```
CALL apoc.schema.assert( {},
{Comic:['name'],Hero:['name']});
```

#### 5. Load the data

Use the following query to pull the data directly from Tomaz's github page

```
CALL apoc.load.csv('https://raw.githubusercontent.com/tomasonjo/neo4j-marvel/master/data/edges.csv') yield map as row WITH row
MERGE (h:Hero {name:row.hero})
MERGE (c:Comic {name:row.comic})
MERGE (h)-[:APPEARS_IN]->(c);
```

## Up Next

Choosing a Graph Model (schema)

## Resources

- [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network)
- [Neo4j Desktop edition](https://neo4j.com/download-center/?ref=web-product-database/#desktop)
- [Graph People blog](https://tbgraph.wordpress.com/)
- [Tomaz Bratanic's github page](https://github.com/tomasonjo)
