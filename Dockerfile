FROM eclipse-temurin:21-jdk

WORKDIR /app
COPY src ./src
COPY index.html ./index.html

RUN javac --add-modules jdk.httpserver -d out src/main/java/vectordb/VectorDbServer.java

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "java --add-modules jdk.httpserver -cp out vectordb.VectorDbServer"]
