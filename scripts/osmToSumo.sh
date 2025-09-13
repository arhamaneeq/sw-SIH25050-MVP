#!/bin/bash
set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input.osm>"
  exit 1
fi

NAME=$1

# Convert OSM to SUMO network
netconvert --osm-files "data/maps/$NAME.osm" \
           --output-file "data/nets/$NAME.net.xml" \
           --tls.guess --tls.join \
           --geometry.remove \
           --roundabouts.guess \
           --ramps.guess

# Generate random trips
python3 $SUMO_HOME/tools/randomTrips.py \
        -n "data/nets/$NAME.net.xml" \
        -o "data/trips/$NAME.trips.xml" \
        -p 5

# Generate routes
duarouter --net-file "data/nets/$NAME.net.xml" \
          --route-files "data/trips/$NAME.trips.xml" \
          --output-file "data/routes/$NAME.rou.xml"
