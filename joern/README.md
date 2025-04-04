joern
====

**This version of joern has been discontinued.**

**Joern lives on at https://github.com/ShiftLeftSecurity/joern**

Source code analysis is full of graphs: abstract syntax trees, control
flow graphs, call graphs, program dependency graphs and directory
structures, to name a few. Joern analyzes a code base using a robust
parser for C/C++ and represents the entire code base by one large
property graph stored in a Neo4J graph database. This allows code to
be mined using complex queries formulated in the graph traversal
languages Gremlin and Cypher.

The documentation can be found [here](http://joern.readthedocs.org/en/latest/)

./joern

## how to build

first install java 1.8 and gradle 4.4.1 (https://services.gradle.org/distributions/gradle-4.4.1-bin.zip)

```
sudo apt update
sudo apt install python3-setuptools libgraphviz-dev
sudo ./build.sh
```

## how to use

```
./joern-parse <output dir> <intput dir>
```

**do not make output dir first!**