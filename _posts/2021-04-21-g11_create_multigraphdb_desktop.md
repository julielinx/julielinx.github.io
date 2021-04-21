---
title: "Entry G11: Create the Marvel Multigraph Database"
categories:
  - Blog
tags:
  - graph
  - neo4j
  - graph analytics
---

Creating a multigraph database was actually way easier than I expected.

I'm still using the [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network) and the directions for creating the initial database are exactly the same as the directions from [Entry G2](https://julielinx.github.io/blog/g02_create_graphdb_desktop/). For easy reference, I'll include most the steps, directions, and pictures, but some may be more condensed. I'm also assuming you've got the Neo4j Desktop downloaded (if not, see [Entry G2](https://julielinx.github.io/blog/g02_create_graphdb_desktop/))

## Create an Empty Database

### 1. Click the `Add` button in the upper right of the My Project area

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/dbadd_button1.png?raw=true'>

### 2. Choose `Local DBMS`

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/dbadd_button2.png?raw=true'>

### 3. Give your database a name and set a password and version

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/db_create.png?raw=true'>

### 4. Admire your new database

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/create_directions.png?raw=true'>

# Setup the database

Just like before, we need to prepare the database for data.


### 1. Click on your new database

This opens the Options for the database. It defaults to the Details page.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/options_default.png?raw=true'>

### 2. Navigate to `Plugins`

We want to add some plugins to make working with the data easier. To do this, choose the `Plugins` tab. You'll see the four available plugins

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/options_plugins.png?raw=true'>

### 3. Install the desired plugins
   
1. Expland the APOC and Graph Data Science libraries individually
2. Click the `Install` button for each

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/options_installplugins.png?raw=true'>

# Fire up the database

### 1. Start the database

Click `Start` to start the database

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data1.png?raw=true'>

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data2.png?raw=true'>

### 2. Open the database

Click `Open` to fire up the browser based interface

You'll notice that once the database is running, the options change slightly; `Start` has changed to `Stop` and `Open` is now selectable.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data3.png?raw=true'>

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data4.png?raw=true'>

### 3. Navigate to Neo4j Browser

You should now have a Neo4j Desktop Browser window open. The initial page usually looks something like this (I'm using Neo4j Desktop version 1.4.1 and database version 4.1.3)

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/import_data5.png?raw=true'>

# Create Multiple Graphs

This is where things start to change. The code in the G11 notebook starts here.

The [Managing Multiple Databases in Neo4j](https://neo4j.com/developer/manage-multiple-databases/) tutorial in the Neo4j [Developer Guides](https://neo4j.com/developer/) has really nice, easy to follow directions on how to create multigraph databases in Neo4j. Here are the directions on how to use those directions for our purposes. 

### 1. Access the `system` database

A Neo4j database is created with two default databases:

- the default `neo4j` database
- a `system` database

The default `neo4j` database is the one we've been using. It's the standard database that holds graph data.

The `system` database holds information about the databases in the Neo4j instance. It is also where we create/edit/delete graphs within the instance.

To access this database type `:use system` into the Neo4j command line:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/use_system.png?raw=true'>

You can type `show databases` to see the two existing databases:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/show_dbs_default.png?raw=true'>

### 2. Create a database for each model

Creating a new, empty database is really easy, just type `create database db_name` into the Neo4j command line. Of course, you'll need to specify the actual name of the database:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/create_db_bimodal.png?raw=true'>

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/create_db_mixmodal.png?raw=true'>

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/create_db_unimodal.png?raw=true'>

# Populate the Bimodal Graph

### 1. Go to the Bimodal Graph

This is much easier than Starting and Stopping graphs in the Project area of Neo4j Desktop. Just enter the following into the Neo4j command line:

`:use db_name`

If you used the same naming convention I did, then it will look like this:

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/use_bimodal.png?raw=true'>

You are now in the bimodal graph.

### 2. Create the schema

Now we can use the exact same instructions from [Entry G2](https://julielinx.github.io/blog/g02_create_graphdb_desktop/) to load the data. 

Input the following in the Neo4j command line to set the schema:
  
```
CALL apoc.schema.assert( {},
{Comic:['name'],Hero:['name']});
```

### 3. Load the data

Pull the data directly from Tomaz's github page into the graph:

```
CALL apoc.load.csv('https://raw.githubusercontent.com/tomasonjo/neo4j-marvel/master/data/edges.csv') yield map as row WITH row
MERGE (h:Hero {name:row.hero})
MERGE (c:Comic {name:row.comic})
MERGE (h)-[:APPEARS_IN]->(c);
```

# Populate the Mixed Graph

This is even easier than cloning the graph like we did in [Entry G5](https://julielinx.github.io/blog/g05_project_bimodal/).

Let's condense the directions from populating the bimodal graph above and Entry G5:

### 1. Enable Multiple Statements

Mine was on by default, but make sure that "Enable multi statement query editor" is checked in the Browser Settings (the gear icon):

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/enable_multistatements.png?raw=true'>

### 2. Go to the Mixed Graph

If you used the same naming convention I did, the code will look like this:

`:use mixmodal`

### 3. Set schema, load data, and project unimodal model

We can combine our statements and let Neo4j run them all one after another:

```
CALL apoc.schema.assert( {},
{Comic:['name'],Hero:['name']});

CALL apoc.load.csv('https://raw.githubusercontent.com/tomasonjo/neo4j-marvel/master/data/edges.csv') yield map as row WITH row
MERGE (h:Hero {name:row.hero})
MERGE (c:Comic {name:row.comic})
MERGE (h)-[:APPEARS_IN]->(c);

CALL apoc.periodic.iterate('MATCH (h1:Hero)-->(:Comic)<--(h2:Hero) where id(h1) < id(h2) RETURN h1, h2',
'MERGE (h1)-[r:KNOWS]-(h2) on CREATE SET r.weight = 1 on MATCH SET r.weight = r.weight+1', {batchSize:5000, parallel:false, iterateList:True});
```

*Side note*, I put an empty line between the different statements to make it clear what code belongs to which statement. However, there is no need for this.

# Populate the Unimodal Graph

Last but not least, we create our unimodal model graph.

### 1. Go to the Unimodal Graph

If you used the same naming convention I did, the code will look like this:

`:use unimodal`

### 3. Set schema, load data, project unimodal model, and delete bimodal elements

We use the same statements as for the mixed model, then add a statement to remove the Comic nodes and their relationships:

```
CALL apoc.schema.assert( {},
{Comic:['name'],Hero:['name']});

CALL apoc.load.csv('https://raw.githubusercontent.com/tomasonjo/neo4j-marvel/master/data/edges.csv') yield map as row WITH row
MERGE (h:Hero {name:row.hero})
MERGE (c:Comic {name:row.comic})
MERGE (h)-[:APPEARS_IN]->(c);

CALL apoc.periodic.iterate('MATCH (h1:Hero)-->(:Comic)<--(h2:Hero) where id(h1) < id(h2) RETURN h1, h2',
'MERGE (h1)-[r:KNOWS]-(h2) on CREATE SET r.weight = 1 on MATCH SET r.weight = r.weight+1', {batchSize:5000, parallel:false, iterateList:True});

MATCH (c:Comic)
DETACH DELETE c;
```

# Check results

We can now see all of our databases have been created by switching back to the `system` database (to see the results of `show database` you will need to run these lines separately, not as multiple statements).

`:use system`

`show databases`

<img src='https://github.com/julielinx/datascience_diaries/blob/master/graph/images/show_dbs_models.png?raw=true'>

That's it for creating the databases. So far I'm finding this much nicer than having each model in its own separate instance. I can now easily switch between graph models while remaining in a Jupyter notebook. This will allow me to group topics much more logically instead of running the code for several topics in a single notebook and having one for each graph model.

## Up Next

Cross Graph Model Degree Comparison

## Resources

- [Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network)
- [Graph People blog](https://tbgraph.wordpress.com/)
- [Tomaz Bratanic's github page](https://github.com/tomasonjo)
- [Managing Multiple Databases in Neo4j](https://neo4j.com/developer/manage-multiple-databases/)
- [Neo4j Developer Guides](https://neo4j.com/developer/)
- [Entry G2: Create a Neo4j Database](https://julielinx.github.io/blog/g02_create_graphdb_desktop/)
- [Entry G5: Projecting Bimodal to Unimodal](https://julielinx.github.io/blog/g05_project_bimodal/)
