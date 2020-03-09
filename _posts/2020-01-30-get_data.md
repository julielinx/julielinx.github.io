---
title: "Entry 4: Get the Data"
categories:
  - Blog
tags:
  - machine learning process
  - get data
---

# Entry 4 - Get the Data

In Entry 3, I defined my problem as:

**Holding all other factors constant, what mass is needed to retain an atmosphere on Mars?**

## The Problem

This sounds like a dataset I'm going to have to create myself.

*[Hands on Machine Learning with Scikit-Learn & TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)* recommends automating as much of the data acquisition process as possible, but this is a one-off dataset and the known parameters of planets doesn't change very often, so I'm not going to worry about that in this entry. If I were working on a project where I would connect to the data source again, like a twitter NLP project, I would most certainly want to automate pulling data. Sounds like a project for another dairy entry on a different mini-project.

If I were going to do any automation of this dataset, it would revolve around scraping table data from an HTML page. 

## The Options

The type of entities from which to draw the necessary data is rather limited. There are the planets, moons, and dwarf planets of this solar system and possibly exoplanets of other systems.

## The Proposed Solution

Fortunately, I didn't have to comb through information on each and every planetary body individually. The [planetary fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/) has many of the features I need. This included 8 planets, 1 moon, and 1 dwarf planet. Starting with this as a base, I gathered 27 features on 11 planetary bodies.

I considered including more moons (Jupiter has 79, Saturn 82, Uranus 27, and Neptune 14), but couldn't find sufficient information on the necessary features. Most importantly was a lack of information on atmospheric mass.
The same was true for exoplanets. They're just too far away to have good measurements.