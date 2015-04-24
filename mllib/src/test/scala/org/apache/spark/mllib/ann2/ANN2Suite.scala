package org.apache.spark.mllib.ann2

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite


class ANN2Suite extends FunSuite with MLlibTestSparkContext {
//
//  test("ANN learns XOR function") {
//    val inputs = Array[Array[Double]](
//      Array[Double](0, 0),
//      Array[Double](0, 1),
//      Array[Double](1, 0),
//      Array[Double](1, 1)
//    )
//    val outputs = Array[Double](0, 1, 1, 0)
//    val data = inputs.zip(outputs).map { case (features, label) =>
//      (Vectors.dense(features), Vectors.dense(Array(label)))
//    }
//    val rddData = sc.parallelize(data, 1)
//    val hiddenLayersTopology = Array[Int](5)
//    val dataSample = rddData.first()
//    val layerSizes = dataSample._1.size +: hiddenLayersTopology :+ dataSample._2.size
//    val topology = Topology.multiLayerPerceptron(layerSizes, true)
//    val initialWeights = FeedForwardModel(topology, 23124).weights()
//    val trainer = new FeedForwardTrainer(topology, 2, 1)
//    trainer.SGDOptimizer.setNumIterations(2000)
//    val model = trainer.train(rddData)
//    //val model = FeedForwardTrainer.train(rddData, 1, 20, topology, initialWeights)
//    val predictionAndLabels = rddData.map { case (input, label) =>
//      (model.predict(input)(0), label(0))
//    }.collect()
//    predictionAndLabels.foreach(println)
//    //assert(predictionAndLabels.forall { case (p, l) => (math.round(p) - l) == 0})
//  }

  test("ANN learns XOR function with coded output") {
    val inputs = Array[Array[Double]](
      Array[Double](0, 0),
      Array[Double](0, 1),
      Array[Double](1, 0),
      Array[Double](1, 1)
    )
    val outputs = Array[Array[Double]](
      Array[Double](1, 0),
      Array[Double](0, 1),
      Array[Double](0, 1),
      Array[Double](1, 0)
    )
    val data = inputs.zip(outputs).map { case (features, label) =>
      (Vectors.dense(features), Vectors.dense(label))
    }
    val rddData = sc.parallelize(data, 1)
    val hiddenLayersTopology = Array[Int](5)
    val dataSample = rddData.first()
    val layerSizes = dataSample._1.size +: hiddenLayersTopology :+ dataSample._2.size
    val topology = Topology.multiLayerPerceptron(layerSizes, true)
    val initialWeights = FeedForwardModel(topology, 23124).weights()
    val trainer = new FeedForwardTrainer(topology, 2, 2)
    trainer.SGDOptimizer.setNumIterations(2000)
    trainer.setWeights(initialWeights)
    val model = trainer.train(rddData)
    //val model = FeedForwardTrainer.train(rddData, 1, 20, topology, initialWeights)
    val predictionAndLabels = rddData.map { case (input, label) =>
      (model.predict(input), label)
    }.collect()
    predictionAndLabels.foreach(println)
    //assert(predictionAndLabels.forall { case (p, l) => (math.round(p) - l) == 0})
  }

}
