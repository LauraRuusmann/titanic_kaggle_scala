package main

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object Titanic {

  val sqlContext: SparkSession = SparkSession
    .builder()
    .appName("Titanic")
    .config("spark.master", "local")
    .getOrCreate()

  def computeAverageAge(training: DataFrame, test: DataFrame): Double = {
    training.select("Age")
      .union(test.select("Age"))
      .agg(avg("Age"))
      .head().getDouble(0)
  }

  def computeAverageFare(training: DataFrame, test: DataFrame): Double = {
    training.select("Fare")
      .union(test.select("Fare"))
      .agg(avg("Fare"))
      .head().getDouble(0)
  }


  def main(args: Array[String]) {

    var training = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("src/main/resources/data/titanic/train.csv")

    var test = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("src/main/resources/data/titanic/test.csv")

    // fill age na-s
    val averageAge = computeAverageAge(training, test)
    training = training.na.fill(averageAge, Array("Age"))
    test = test.na.fill(averageAge, Array("Age"))

    // fill embarked na-s
    training = training.na.fill("S", Array("Embarked"))
    test = test.na.fill("S", Array("Embarked"))

    // fill fare na-s
    val averageFare = computeAverageFare(training,test)
    training = training.na.fill(averageFare, Array("Fare"))
    test = test.na.fill(averageFare, Array("Fare"))

    // make title categories more general
    val nameToTitle: String => String = _.replaceAll(""".*, |\..*""", "") match {
      case "Miss" => "Miss"
      case "Mr" => "Mr"
      case "Mrs" => "Mrs"
      case "Lady"   => "Mrs"
      case "Mme"    => "Mrs"
      case "Ms"     => "Ms"
      case "Miss"   => "Miss"
      case "Mlle"   => "Miss"
      case "Don"    => "Mr"
      case "Sir"    => "Mr"
      case "Col"    => "Col"
      case "Capt"   => "Col"
      case "Major"  => "Col"
      case _ => "RareTitle"

    }

    training = training.withColumn("title", udf(nameToTitle).apply(training("Name")))
    test = test.withColumn("title", udf(nameToTitle).apply(test("Name")))

    //whether the person is a mother
    val mother = when(expr("Sex == 'female' AND Age > 18.0 AND Parch > 1"), 1).otherwise(0)

    training = training.withColumn("mother", mother)
    test = test.withColumn("mother", mother)

    //spark.ml can’t handle categorical features or labels unless they are indexed

    // Index Sex
    val sexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndex")
      .fit(training)

    // Index Embarked
    val embarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndex")
      .fit(training)

    // Index Title
    val titleIndexer = new StringIndexer()
      .setInputCol("title")
      .setOutputCol("titleIndex")
      .fit(training)

    // Index Pclass
    val classIndexer = new StringIndexer()
      .setInputCol("Pclass")
      .setOutputCol("pclassIndex")
      .fit(training)

    // Index Survived as the Label
    val labelIndexer = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("Label")
      .fit(training)

    val features: Array[String] = Array("pclassIndex", "Age", "Fare", "SexIndex", "EmbarkedIndex", "titleIndex", "mother")

    //assembling all feature columns into one vector column because every spark.ml machine learning algorithm expects that
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("Features")

    val randomForest = new RandomForestClassifier()
      .setLabelCol("Label")
      .setFeaturesCol("Features")

    //A pipeline is an ordered combination of Transformers and Estimators.
    val pipeline = new Pipeline().setStages(
      Array(sexIndexer,
        embarkedIndexer,
        titleIndexer,
        labelIndexer,
        classIndexer,
        assembler,
        randomForest
      )
    )

    // training the model
    val model = pipeline.fit(training)

    // using the model to make predictions
    val predictions = model.transform(test)
    predictions.selectExpr("PassengerId", "cast(prediction as int) Survived")
      .repartition(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save("target/predictions.csv")


    // With cross validation
    val params = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth, Array(4, 8, 12))
      .addGrid(randomForest.numTrees, Array(15, 30, 50))
      .addGrid(randomForest.maxBins, Array(16, 32, 64))
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderPR")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(params)
      .setNumFolds(10)

    val crossValidatorModel = cv.fit(training)
    val predictionsCV = crossValidatorModel.transform(test)

    predictionsCV.selectExpr("PassengerId", "cast(prediction as int) Survived")
      .repartition(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save("target/predictionsCV.csv")

  }

}
