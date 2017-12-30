name := "data_mining_scala"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0" % Provided
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0" % Compile
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0" % Compile
libraryDependencies += "com.databricks" %% "spark-csv" % "1.2.0" % Compile
