#!/bin/bash
set -e

# Check input
if [ $# -ne 1 ]; then
    echo "Usage: $0 <map_name_without_ext>"
    exit 1
fi

NAME=$1

# Ensure required directories exist
mkdir -p data/nets data/trips data/routes data/sumocfgs

# --- Linux paths (for SUMO tools in WSL) ---
NET_PATH_LIN=$(realpath "data/nets/$NAME.net.xml")
TRIP_PATH_LIN=$(realpath "data/trips/$NAME.trips.xml")
ROUTE_PATH_LIN=$(realpath "data/routes/$NAME.rou.xml")
SUMOCFG_FILE_LIN=$(realpath "data/sumocfgs/$NAME.sumocfg")

# --- Windows paths (for .sumocfg consumed by Windows SUMO) ---
NET_PATH_WIN=$(wslpath -w "$NET_PATH_LIN")
TRIP_PATH_WIN=$(wslpath -w "$TRIP_PATH_LIN")
ROUTE_PATH_WIN=$(wslpath -w "$ROUTE_PATH_LIN")

echo ">=> Converting OSM to SUMO network..."
netconvert --osm-files "data/maps/$NAME.osm" \
           --output-file "$NET_PATH_LIN" \
           --tls.guess --tls.join \
           --geometry.remove \
           --roundabouts.guess \
           --ramps.guess

echo ">=> Generating random trips..."
python3 $SUMO_HOME/tools/randomTrips.py \
        -n "$NET_PATH_LIN" \
        -o "$TRIP_PATH_LIN" \
        -p 5 \
        --random

echo ">=> Generating routes from trips..."
duarouter --net-file "$NET_PATH_LIN" \
          --route-files "$TRIP_PATH_LIN" \
          --output-file "$ROUTE_PATH_LIN" \
          --ignore-errors true

echo ">=> Generating SUMO configuration file..."
cat > "$SUMOCFG_FILE_LIN" <<EOL
<configuration>
    <input>
        <net-file value="$NET_PATH_WIN"/>
        <route-files value="$ROUTE_PATH_WIN"/>
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
echo "SUMO configuration ready: $SUMOCFG_FILE_LIN"
