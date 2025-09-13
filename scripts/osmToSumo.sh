# !/bin/bash

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <input.osm> <output.net.xml>"
  exit 1
fi

INPUT_OSM=$1
OUTPUT_NET=$2

netconvert --osm-files "$INPUT_OSM" \
           --output-file "$OUTPUT_NET" \
           --tls.guess --tls.join \
           --geometry.remove \
           --roundabouts.guess \
           --ramps.guess