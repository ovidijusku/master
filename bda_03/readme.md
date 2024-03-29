## Pre-start
Make sure that Docker volume storage is clean before launching the application

```
docker volume rm $(docker volume ls -q)
``` 

## Launch

1. Copy `csv` file into `task_3` directory, all following commands should be executed from `task_3` directory.
2. Start mongodb sharded cluster
```
docker-compose -f cluster-docker-compose.yml up -d
```
3. Start app with data insert task. In order to not overwhelm memory, only first 1M rows are taken into account. Launching this task will also create database, collection and will enable sharding for them based on the vessel id. Insert chunk size was selected 1024 as it was the fastest and not overwhelming IO database operations. In comparison, when using chunk sizes higher than 1024, there were cases when shard replicas were killed.
```
docker-compose -f insert-docker-compose.yml up --build
```

Note: if you run command immediately after command from 2nd step, you will get error:
```
pymongo.errors.OperationFailure: Database sharded_cluster could not be created :: caused by :: No shards found
```
In order to prevent that, give some time (approx 30 seconds) for the sharded cluster to setup everything.

4. Start data extraction script. This will firstly setup indexing on these columns: `MMSI`, `ROT`, `SOG`, `COG`, `heading`. Then it extracts raw data from raw collection and inserts that to the processed collection based on multiple relevant filters. New collection gets indexing on the `MMSI` and `timestamp` columns. Based on the sorting conditions, data is extracted from new collection and based on `MMSI`, time differences in milliseconds are extracted for each vessel. Results are later presented in histograms which are stored in `histogram/` directory. Each histogram has same logarithmic time difference values (x-axis) and logarithmic frequencies (y-axis) in order to spot outliers more easily.
```
docker-compose -f extract-docker-compose.yml up --build
```

## Insights
When looking at the histograms, majority of them appear to be either normal. Smaller portion of them are right-skewed or left-skewed.

1. Moving and moored vessels have different transmitting times. F.e. MMSI `215760000`
2. Vessels when being in slow paced activities (`Engaged in fishing` or `Restricted maneuverability`) also increase the time intervals. F.e. MMSI `253338000`


## References

Mongodb sharded cluster setup. Used Github [repository](https://github.com/pkdone/sharded-mongodb-docker ) as an example. 
