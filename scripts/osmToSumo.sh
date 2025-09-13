#!/bin/bash
set -e

# Check input
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input.osm>"
    exit 1
fi

NAME=$1

# Ensure required directories exist
mkdir -p data/nets data/trips data/routes data/sumocfgs

echo ">=> Converting OSM to SUMO network..."
NET_PATH=$(realpath "data/nets/$NAME.net.xml")
netconvert --osm-files "data/maps/$NAME.osm" \
           --output-file "$NET_PATH" \
           --tls.guess --tls.join \
           --geometry.remove \
           --roundabouts.guess \
           --ramps.guess

echo ">=> Generating random trips..."
TRIP_PATH=$(realpath "data/trips/$NAME.trips.xml")
python3 $SUMO_HOME/tools/randomTrips.py \
        -n "$NET_PATH" \
        -o "$TRIP_PATH" \
        -p 5 \
        --random

echo ">=> Generating routes from trips..."
ROUTE_PATH=$(realpath "data/routes/$NAME.rou.xml")
duarouter --net-file "$NET_PATH" \
          --route-files "$TRIP_PATH" \
          --output-file "$ROUTE_PATH" \
          --ignore-errors true

echo ">=> Generating SUMO configuration file..."
SUMOCFG_FILE=$(realpath "data/sumocfgs/$NAME.sumocfg")
cat > "$SUMOCFG_FILE" <<EOL
<configuration>
    <input>
        <net-file value="$NET_PATH"/>
        <route-files value="$ROUTE_PATH"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
EOL

echo "<=> All done!"
echo "SUMO configuration ready: $SUMOCFG_FILE"
