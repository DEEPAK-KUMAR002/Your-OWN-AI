#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
mkdir -p out
javac --add-modules jdk.httpserver -d out src/main/java/vectordb/VectorDbServer.java
exec java --add-modules jdk.httpserver -cp out vectordb.VectorDbServer
