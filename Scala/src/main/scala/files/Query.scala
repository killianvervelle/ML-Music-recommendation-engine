package files

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.{ClusteringEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, Map}


object Query {

    val spark: SparkSession = SparkSession.builder
      .appName("Test")
      .master("local[*]")
      .config("spark.sql.debug.maxToStringFields", "1000000")
      .getOrCreate()

    private val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")

    // Specify the path to your Parquet file
    val parquetFile = "./src/main/data/data.parquet"


    // Functions
    def main(args: Array[String]): Unit = {
        val data: DataFrame = read_file()
        // question 1: which year holds the highest number of produced tracks ?
        val groupedYear = groupByCount(data,"year").filter(col("year") =!= 0)
        groupedYear.show()
        // which country is home to the highest number of artists ?
        val groupedLocation = groupByCount(data,"artist_location")
        groupedLocation.show()
        // which are the most popular music genre ?
        val groupedGenre = uniqueGenreCount(data, columnName = "artist_terms")
        groupedGenre.show()

        // question 2: what is the average BPM per music genre?
        val listGenre = groupedGenre.select("term").collect().map(_.getString(0)).toList
        val avgBPM = avgMetricbyGenre(data, "tempo", listGenre)
        printResults(avgBPM, "tempo")
        // what is the average loudness per music genre?
        val avgLoudness = avgMetricbyGenre(data, "loudness", listGenre)
        printResults(avgLoudness, "loudness")

        // question 3:
        // Data transformation
        val scaledFeatures = preprocessing(data)
        scaledFeatures.show(truncate = false)
        // Model definition and prediction
        multilayerperceptron(scaledFeatures)
        DecisionTree(scaledFeatures)
        randomforest(scaledFeatures)

        // question 3: how to predict the music genre from features like dB, tempo, scale etc.
        // Select meaningful columns for the clustering
        val features_for_kmeans = Array("duration", "key", "loudness", "tempo", "time_signature")
        val scaledData = preprocess_data(data, features_for_kmeans)

        // Find the best silhouettes
        val minClusters: Int = 2
        val maxClusters: Int = 12
        val silhouettes = get_silhouettes(scaledData, minClusters, maxClusters)
        println(s"Silhouettes with squared euclidean distance :")
        println(silhouettes.mkString(", "))

        // Trains a k-means model.
        // Get the number of clusters which yields the maximum silhouette
        val maxVal: Double = silhouettes.max
        val nClusters: Int = silhouettes.indexOf(maxVal) + minClusters
        val predictions = kmeans_predict_show(scaledData, nClusters, features_for_kmeans)

        // Print the song of a playlist
        val features_to_show = Array("artist_name", "title", "duration", "tempo", "artist_genre")
        show_predicted_musics(data, predictions, 0, 20, features_to_show)
        show_predicted_musics(data, predictions, 2, 20, features_to_show)

        // question 4:
        // Dans une optique de recommandation d'un artiste à un utilisateur,
        // comment pourrait-on mesurer la similarité entre artistes ?
        question4(data)
    }

    private def preprocess_data(data: DataFrame, columns: Array[String]): DataFrame = {
        // Select meaningful columns for the clustering
        val dataset = data.select(columns.head, columns.tail: _*)
        println(s"Selected data for k-means:")
        dataset.show(10)

        // Define the assembler
        val assembler = new VectorAssembler()
          .setInputCols(columns)
          .setOutputCol("features")

        // Transform the DataFrame using the VectorAssembler
        val assembledData = assembler.transform(dataset)

        // Create a StandardScaler instance
        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithMean(true) // Optionally remove the mean from the feature vector
          .setWithStd(true) // Optionally scale the features to unit standard deviation

        // Compute summary statistics and generate the scaler model
        val scalerModel = scaler.fit(assembledData)

        // Transform the DataFrame to apply scaling
        val scaledData: DataFrame = scalerModel.transform(assembledData)
        scaledData
    }
    private def kmeans_prediction(data: DataFrame, nClusters: Int): DataFrame = {
        // Trains a k-means model.
        val kmeans = new KMeans().setK(nClusters)
          .setFeaturesCol("scaledFeatures")
          .setSeed(1L)
        val model = kmeans.fit(data)

        // Make predictions
        val predictions = model.transform(data)
        predictions
    }

    private def kmeans_predict_show(data: DataFrame, nClusters: Int, features_for_kmeans: Array[String]): DataFrame = {
        // Trains a k-means model.
        val kmeans = new KMeans().setK(nClusters)
          .setFeaturesCol("scaledFeatures")
          .setSeed(1L)
        val model = kmeans.fit(data)

        // Make predictions
        val predictions = model.transform(data)

        // Shows the result.
        println(s"Cluster Centers for nClusters = $nClusters : ")
        println(features_for_kmeans.mkString(", "))
        model.clusterCenters.foreach(println)
        predictions
    }

    private def get_silhouettes(data: DataFrame, minClusters: Int, maxClusters: Int): ArrayBuffer[Double]  = {
        // Loop and store results
        val silhouettes = ArrayBuffer[Double]() // Mutable collection to store results

        for (nClusters <- minClusters to maxClusters) {
            val predictions = kmeans_prediction(data, nClusters) // Call the function with the current value
            // Evaluate clustering by computing Silhouette score
            val evaluator = new ClusteringEvaluator()

            val silhouette = evaluator.evaluate(predictions)
            silhouettes += silhouette // Store the result in the collection
        }
        silhouettes
    }

    private def show_predicted_musics(data: DataFrame, predictions: DataFrame, cluster_id: Int, musics_to_show: Int, features_to_print: Array[String]) : Unit = {
        // Add row index to data
        val dataWithIndex = data.withColumn("index", monotonically_increasing_id())
        // Add row index to predictions
        val predictionCol = predictions.select("prediction")
        val predictionsWithIndex = predictionCol.withColumn("index", monotonically_increasing_id())
        val filteredSongsDF = dataWithIndex.join(predictionsWithIndex, Seq("index"))
          .filter(col("prediction") === cluster_id)
        val filteredSongsFeatures = filteredSongsDF.select(features_to_print.head, features_to_print.tail: _*)
        println(s"First $musics_to_show musics of cluster $cluster_id :")
        filteredSongsFeatures.show(musics_to_show, false)
    }

    def read_file(): DataFrame = {
        // Read the Parquet file into a DataFrame and sort it
        val data: DataFrame = Data.readParquetFile(parquetFile)
        // Get the unique values of the feature "year" and remove rows where year = 0
        data
    }

    def groupByCount(data: DataFrame, columnName: String): DataFrame = {
        // remove rows where columnName == 0
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull)
        // sort the dataframe on columnName
        val sortedData: DataFrame = filteredData.sort(col(columnName).asc)
        // group the dataframe
        val groupedData = sortedData.groupBy(columnName).count().sort(col("count").desc)
        groupedData
    }

    def groupByMean(data: DataFrame, columnName: String): DataFrame = {
        // remove rows where columnName == 0 or Nan
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull && !isnan(col(columnName)) && col(columnName) =!= 0)
        // sort the dataframe on columnName
        val sortedData: DataFrame = filteredData.sort(col(columnName).asc)
        // group the dataframe by "year"
        val groupedData = sortedData.groupBy(columnName).mean().sort(col("mean").desc)
        groupedData
    }

    def avgMetricbyGenre(data: DataFrame, columnName: String, listGenre: List[String]): mutable.Map[String, Double] = {
        val top10Genres = listGenre.take(10).distinct.map(_.trim).filter(_.nonEmpty).map(_.trim.toLowerCase)
        val dfWithArrayGenre: DataFrame = data.withColumn("artist_terms_array", split(trim(col("artist_terms"), "[]"), ","))
        val result = scala.collection.mutable.Map[String, Double]() // Mutable map to store results

        // Filter the DataFrame based on the top 10 genres and calculate the average metric
        for (term <- top10Genres) {
            var count = 0
            var metricTotal = 0.0
            val rows = dfWithArrayGenre.collect()
            rows.foreach { row =>
                val terms = row.getAs[mutable.WrappedArray[String]]("artist_terms_array").map(_.trim.toLowerCase)
                val metric = row.getAs[Double](columnName)
                if (terms.contains(term)) {
                    count += 1 // Increment count if the row contains the term
                    metricTotal += metric
                }
            }
            val avgMetric: Double = if (count > 0) metricTotal / count else 0.0 // Calculate the average metric by dividing the total by the count
            result(term) = avgMetric
        }
        result
    }

    def printResults(result: Map[String, Double], columnName: String): Unit = {
        result.toMap // Convert mutable map to immutable map and return
        result.foreach { case (term, mean) =>
            println(s"Music genre: $term", f" Average ${columnName}: {$mean}")
        }
    }

    def uniqueGenreCount(data: DataFrame, columnName: String): DataFrame = {
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull)
        val dfExploded = filteredData.select(explode(split(col(columnName), ",")).as("term")) // Split the string into an array using "|"
        val termCounts = dfExploded.groupBy("term").agg(count("*").as("count")).orderBy(col("count").desc)
        val filteredTermCount = termCounts.withColumn("term", regexp_replace(col("term"), "[\\[\\]]", ""))

        filteredTermCount
    }

    def preprocessing(dataFrame: DataFrame): DataFrame = {
        // Select the columns of interest
        val selectedDF: DataFrame = dataFrame.select("loudness", "tempo", "time_signature", "beats_start", "duration", "artist_terms")

        // Print the number of NaN values per column
        NanCount(selectedDF)

        // Filter out the NaN values and check dataframe
        val filteredDF: DataFrame = selectedDF.filter(!selectedDF.columns.map(colName => isnan(selectedDF(colName)) || isnull(selectedDF(colName))).reduce(_ || _))
        NanCount(filteredDF)

        // Remove music with time signature = 0
        val filteredSignature: DataFrame = filteredDF.filter(col("time_signature") =!= 0)

        // Check if rows were removed
        //uniqueVal(filteredSignature, "time_signature")

        // remove music with tempo = 0
        val filteredtempo: DataFrame = filteredSignature.filter(col("tempo") =!= 0)

        // Check if rows were removed
        //uniqueVal(filteredtempo, columnName = "tempo")

        // round the numbers in the duration column
        val rounded_duration: DataFrame = filteredtempo.withColumn("duration", round(col("duration")).cast("int"))

        // round the numbers in the tempo column
        val rounded_tempo: DataFrame = rounded_duration.withColumn("tempo", round(col("tempo")).cast("int"))
        //uniqueVal(rounded_tempo, columnName = "tempo")

        // add constant=100 to Loudness to remove skewness
        val addCst: DataFrame = rounded_tempo.withColumn(colName = "loudness", round(col("loudness") + 100).cast("int"))
        //uniqueVal(addCst, columnName = "loudness")

        // Min/max scaling of the features
        val scaledFeatures = minmaxScaling(addCst)

        scaledFeatures

    }

    def NanCount(dataFrame: DataFrame) : Unit = {
        val nanCountsPerColumn = dataFrame.columns.map(colName => dataFrame.filter(col(colName).isNull || col(colName).isNaN).count())
        dataFrame.columns.zip(nanCountsPerColumn).foreach { case (colName, count) =>
            // println(s"Column '$colName' has $count NaN value(s).")
        }
    }

    def uniqueVal(dataFrame: DataFrame, columnName: String) : Unit =  {
        val uniqueValues: Array[Any] = dataFrame.select(columnName).distinct().collect().map(_.get(0))
        uniqueValues.foreach(println)
    }

    def minmaxScaling(dataFrame: DataFrame): DataFrame = {
        // Create a sample DataFrame with three integer columns
        val selected: DataFrame = dataFrame.select(col = "loudness", "tempo", "duration", "time_signature", "artist_terms")
        val inputCols = Array("loudness", "tempo", "duration")

        // Combine integer columns into a vector column
        val assembler = new VectorAssembler()
          .setInputCols(inputCols)
          .setOutputCol("features")
        val assembledData: DataFrame = assembler.transform(selected)

        // Create a MinMaxScaler instance
        val scaler = new MinMaxScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")

        // Compute the min-max scaling model
        val scalerModel: MinMaxScalerModel = scaler.fit(assembledData)

        // Apply the scaling to the data
        val scaledData: DataFrame = scalerModel.transform(assembledData)

        // Define a UDF to extract values from the sparse vector
        val extractValues = udf((vector: Vector) => vector.asInstanceOf[DenseVector].values)

        // Create separate columns for each value
        val updatedDF = scaledData
          .withColumn("values", extractValues(col("scaledFeatures")))
          .withColumn("loudness_sc", col("values").getItem(0))
          .withColumn("tempo_sc", col("values").getItem(1))
          .withColumn("duration_sc", col("values").getItem(2))
          .drop("scaledFeatures", "features", "values") // Drop the original 'scaledFeatures' column

        // Filter artiste_genre only on the most popular music genre defined in the previous question
        val topGenres = List("rock", "pop", "electronic", "jazz", "folk").map(_.trim.toLowerCase)

        // Remove the square brackets and single quotes from the artist_genre column
        val cleanedGenre = regexp_replace(col("artist_terms"), "[\\[\\]']", "")

        // Split the cleaned genre column by the comma character (',')
        val splitGenre = split(cleanedGenre, ", ")

        // Remove terms from the splitGenre that are not in topGenres
        val filteredGenre = array_join(array_intersect(splitGenre, typedLit(topGenres)), ",").as("filtered_genre")

        // Replace the existing genre column with the filteredGenre
        val transformedDF: DataFrame = updatedDF
          .withColumn("filtered_genre", filteredGenre)
          .withColumn("filtered_genre", split(col("filtered_genre"), ",")(0))
          .withColumn("loudness_sc", round(col("loudness_sc"), 2))
          .withColumn("tempo_sc", round(col("tempo_sc"), 2))
          .withColumn("duration_sc", round(col("duration_sc"), 2))
        val finalDF: DataFrame = one_hot_encoding(transformedDF.filter(col("filtered_genre") =!= "").drop("artist_terms"), "filtered_genre")
          .withColumn("filtered_genre_index", col("filtered_genre_index").cast("int"))

        finalDF
    }

    def applyTrimAndLowerCase(df: DataFrame, columnName: String): DataFrame = {
        val trimAndLowerCase = udf((arr: Seq[String]) => arr.map(_.trim.toLowerCase))
        df.withColumn(columnName, trimAndLowerCase(col(columnName)))
        df
    }

    def one_hot_encoding(df: DataFrame, columnName: String): DataFrame = {
        // StringIndexer to convert the categorical column to numeric indices
        val indexer = new StringIndexer()
          .setInputCol(columnName)
          .setOutputCol(s"${columnName}_index")
          .fit(df)

        val indexedData = indexer.transform(df)

        // OneHotEncoder to perform one-hot encoding
        val encoder = new OneHotEncoder()
          .setInputCols(Array(s"${columnName}_index"))
          .setOutputCols(Array(s"${columnName}_encoded"))

        // Create a pipeline with StringIndexer and OneHotEncoder
        val pipeline = new Pipeline().setStages(Array(indexer, encoder))

        // Fit the pipeline on the data
        val model = pipeline.fit(df)

        // Transform the data using the pipeline model
        val encodedData = model.transform(df).drop("filtered_genre_encoded")

        encodedData
    }

    def multilayerperceptron(df: DataFrame) : Unit =  {
        val layers = Array[Int](4, 10, 10, 5) // Adjust the number of nodes in each layer
        val maxIter = 50 // Adjust the maximum number of iterations
        val blockSize = 128 // Adjust the block size
        val stepSize = 0.3 // Adjust the learning rate
        val seed = 1234L // Fix the random seed

        // Step 1: Load and split the data into training and testing sets
        val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

        // Step 3: Define the feature transformation
        val featureAssembler = new VectorAssembler()
          .setInputCols(Array("loudness_sc", "tempo_sc", "duration_sc", "time_signature"))
          .setOutputCol("features")

        // Step 4: Select a supervised learning algorithm (Logistic Regression)
        val MLP = new MultilayerPerceptronClassifier()
          .setSeed(seed)
          .setLabelCol("filtered_genre_index")
          .setFeaturesCol("features")
          .setLayers(layers)
          .setMaxIter(maxIter)
          .setBlockSize(blockSize)
          .setStepSize(stepSize)
          .setSolver("l-bfgs") // or "sgd" or "adam"

        // Step 5: Train the model
        // Specify the number of epochs
        val pipeline = new Pipeline().setStages(Array(featureAssembler, MLP))
        val model = pipeline.fit(trainingData)

        // Step 6: Make predictions
        val predictions = model.transform(testData)
        val predictionAndLabels = predictions.select("prediction", "filtered_genre_index")

        // Step 7: Evaluate the model
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("filtered_genre_index")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

        // Calculate and print the accuracy at each epoch
        println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
    }

    def DecisionTree(df: DataFrame): Unit = {
        // Split the data into training and test sets (20% held out for testing).
        val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))
        // Automatically identify categorical features, and index them.
        // Here, we treat features with > 4 distinct values as continuous.
        val assembler = new VectorAssembler()
          .setInputCols(Array("loudness_sc", "tempo_sc", "duration_sc", "time_signature"))
          .setOutputCol("features")

        // Transform the input data to include the assembled feature vector.
        val assembledData = assembler.transform(df)

        // Automatically identify categorical features, and index them.
        // Here, we treat features with > 4 distinct values as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .fit(assembledData)

        // Train a DecisionTree model.
        val dt = new DecisionTreeClassifier()
          .setLabelCol("filtered_genre_index")
          .setFeaturesCol("indexedFeatures")
          .setMaxDepth(20) // Adjust the maxDepth value
          .setMaxBins(32) // Adjust the maxBins value
          .setImpurity("gini") // or "entropy"
          .setMinInstancesPerNode(50) // Adjust the minInstancesPerNode value

        // Chain assembler, indexer, and tree in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(assembler, featureIndexer, dt))

        // Train model. This also runs the indexer.
        val model = pipeline.fit(trainingData)

        // Make predictions.
        val predictions = model.transform(testData)

        // Select example rows to display.
        predictions.select("prediction", "filtered_genre_index", "features").show(5)

        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("filtered_genre_index")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Accuracy on test data = $accuracy")

        val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
        //println(s"Learned regression tree model:\n ${treeModel.toDebugString}")
    }

    def randomforest(df: DataFrame): Unit = {
        // Split the data into training and test sets (20% held out for testing).
        val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))
        // Automatically identify categorical features, and index them.
        // Here, we treat features with > 4 distinct values as continuous.
        val assembler = new VectorAssembler()
          .setInputCols(Array("loudness_sc", "tempo_sc", "duration_sc", "time_signature"))
          .setOutputCol("features")

        // Transform the input data to include the assembled feature vector.
        val assembledData = assembler.transform(df)

        // Automatically identify categorical features, and index them.
        // Here, we treat features with > 4 distinct values as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .fit(assembledData)

        // Train a DecisionTree model.
        val dt = new RandomForestClassifier()
          .setLabelCol("filtered_genre_index")
          .setFeaturesCol("indexedFeatures")
          .setNumTrees(200)
          .setMaxDepth(5) // Adjust the maxDepth value
          .setMaxBins(50) // Adjust the maxBins value
          .setFeatureSubsetStrategy("all") // or "all", "sqrt", "log2", or a fraction value
          .setMinInstancesPerNode(50) // Adjust the minInstancesPerNode value
          .setSubsamplingRate(0.8)

        // Chain assembler, indexer, and tree in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(assembler, featureIndexer, dt))

        // Define the parameter grid (although we are using fixed parameters).
        val paramGrid = new ParamGridBuilder().build()

        // Unquote this section to run the parameter gridsearch. If you do, you need to quote the previous line of code
        //val paramGrid = new ParamGridBuilder()
          //.addGrid(dt.maxDepth, Array(5, 10, 15))
          //.addGrid(dt.maxBins, Array(16, 32, 48))
          //.addGrid(dt.numTrees, Array(50, 100, 200))
          //.build()

        // Set up cross-validation.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("filtered_genre_index")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

        // Unquote this section to run the cross validation validator
        val crossValidator = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(5) // Set the number of folds for cross-validation

        // Unquote this section to run the cross validation and find the best model.
        val cvModel = crossValidator.fit(trainingData)

        // Unquote this section to Print the results at each fold.
        val avgMetrics = cvModel.avgMetrics
        avgMetrics.zipWithIndex.foreach { case (metric, _) =>
            println(s"Model accuracy across all 5 folds: $metric")
        }

        // Unquote this section to get the best model and its parameters.
        //val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
        //val bestParams: Map[Param[_], Any] = Map(bestModel.stages.last.extractParamMap().toSeq.map(paramPair => paramPair.param -> paramPair.value): _*)

        // Unquote this section to print the best params at the end of the gridSearch :
        //println("Best Model Parameters:")
        //bestParams.foreach { case (param, value) =>
            //println(s"${param.name}: $value")
        //}

        // Make predictions on the test data using the best model.
        val predictions = cvModel.transform(testData)

        // Select example rows to display.
        predictions.select("prediction", "filtered_genre_index", "features").show(20)

        val accuracy = evaluator.evaluate(predictions)
        println(s"Accuracy on test data = $accuracy")
    }

    def findMissingValues(df: DataFrame): Unit = {
        val booleanToInt = udf((value: Boolean) => if (value) 1 else 0)
        val missingValues = df.select(df.columns.map(c => sum(booleanToInt(col(c).isNull || col(c).isNaN)).alias(c)): _*)
        missingValues.show()
    }

    def question4(data: DataFrame): Unit = {
        val artistData = create_artist_dataframe(data)

        val test = get_similar_artists(data)
        val scaledData = preprocessData(artistData)
        val predictions = performKMeans(scaledData, 10)
        evaluateCluster(predictions, test, true)

        val ks = Array(5, 10, 20, 50, 100)
        val metrics = ks.map(k => (k, evaluateCluster(performKMeans(scaledData, k), test, false)))
        println(metrics.mkString("\n"))
    }

    def preprocessData(data: DataFrame): DataFrame = {
        // à partir de artistData, on veut filtrer les colonnes pour ne récupérer que celles qui contiennent des float
        val assembler = new VectorAssembler()
          .setInputCols(Array("avgSongDuration", "avgSongLoudness", "avgTempo", "avgLoudness", "avgEnergy"))
          .setOutputCol("features")

        val assembledData = assembler.transform(data)

        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithMean(true)
          .setWithStd(true)

        scaler.fit(assembledData).transform(assembledData)
    }

    def performKMeans(data: DataFrame, k: Int): DataFrame = {
        val kmeans = new KMeans().setK(k)

        val predictions = kmeans.fit(data).transform(data)
        predictions
    }

    def evaluateCluster(predictions: DataFrame, test: DataFrame, verbose: Boolean): Tuple4[Double, Double, Double, Double] = {
        val clusters = predictions.select("prediction", "artist_id").groupBy("prediction").agg(collect_list("artist_id").as("artist_ids"))
        val avgArtistsPerCluster = clusters.select(avg(size(col("artist_ids")))).first().getDouble(0)
        if (verbose) println("avgArtistsPerCluster: " + avgArtistsPerCluster)

        val evaluator = new ClusteringEvaluator()
        val silhouette = evaluator.evaluate(predictions)
        if (verbose) println("Silhouette with squared euclidean distance = " + silhouette)

        val joinedDF = clusters.join(predictions.select("artist_id", "prediction"), Seq("prediction"))
        val removeIdFromSimilar = udf((artist_ids: Seq[String], artist_id: String) => {
            artist_ids.filterNot(_ == artist_id)
        })
        val pred = joinedDF.withColumn("similar", removeIdFromSimilar(col("artist_ids"), col("artist_id"))).select("artist_id", "similar")

        val joinedDF2 = test.join(pred, Seq("artist_id"))

        val calculatePercentage = udf((similarTest: Seq[String], similarPred: Seq[String]) => {
            val commonIds = similarTest.intersect(similarPred).distinct
            val percentage = (commonIds.length.toDouble / similarTest.length) * 100
            percentage
        })

        val resultDF = joinedDF2.withColumn("pourcentage", calculatePercentage(col("similar_artists"), col("similar")))
        val avgPercent = resultDF.select(avg(col("pourcentage"))).first().getDouble(0)
        val maxPercent = resultDF.select(max(col("pourcentage"))).first().getDouble(0)
        if (verbose) {
            println(s"avg accuracy: $avgPercent, max accuracy: $maxPercent")
        }
        Tuple4(avgArtistsPerCluster, silhouette, avgPercent, maxPercent)
    }

    def printNames(predictions: DataFrame): Unit = {
        val names = predictions.select("prediction", "artist_name").groupBy("prediction").agg(collect_list("artist_name").as("similar_artists"))
        val names2 = names.join(predictions.select("artist_name", "prediction"), Seq("prediction"))
        names2.select("artist_name", "similar_artists").show(10, false)
    }

    def create_artist_dataframe(data: DataFrame): DataFrame = {
        val artistData = data.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude", "duration", "energy", "loudness", "tempo", "year")
        val groupedArtist = artistData.groupBy("artist_id")
        val artist_infos = groupedArtist.agg(
            first("artist_name").as("artist_name"),
            count("artist_id").as("nbSong"),
            avg("duration").as("avgSongDuration"),
            avg("loudness").as("avgSongLoudness"),
            avg("tempo").as("avgTempo"),
            min("year").as("yearFirstSong"),
            max("year").as("yearLastSong"),
            avg("loudness").as("avgLoudness"),
            avg("energy").as("avgEnergy")
        )
        artist_infos.na.drop()
    }

    def get_similar_artists(data: DataFrame): DataFrame = {
        val df = data.select("artist_id", "similar_artists").groupBy("artist_id").agg(first("similar_artists").as("similar_artists"))
        val parseStringList = spark.udf.register("parseStringList", (chaine: String) => {
            chaine.drop(1).dropRight(1).split(",").toList.map(_.drop(1).dropRight(1))
        })
        df.withColumn("similar_artists", parseStringList(df("similar_artists")))
    }

}
