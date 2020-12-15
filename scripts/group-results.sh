#!/bin/sh

DESCRIPTION=$1

DATE=$(TZ="Europe/Berlin" date +"%d.%m.%Y-%H.%M")
DIR="$DESCRIPTION-$DATE"

mkdir $DIR

[ -f nvbitfi-igprofile.txt ] && mv nvbitfi-igprofile.txt $DIR
mv  results-* $DIR
mv ../results/* $DIR

[ -d injection-list ] && mv injection-list $DIR
[ -d sdcs ] && mv sdcs $DIR
mv fastWalshTransform* $DIR
