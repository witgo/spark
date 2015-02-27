/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import org.apache.spark.mllib.ann2.{Topology, FeedForwardNetwork, FeedForwardModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.{argmax => Bargmax}

import scala.util.Random

trait ANN2ClassifierHelper {

  protected val labelToIndex: Map[Double, Int]
  private val indexToLabel = labelToIndex.map(_.swap)
  private val labelCount = labelToIndex.size

  protected def labeledPointToVectorPair(labeledPoint: LabeledPoint) = {
    val output = Array.fill(labelCount){0.1}
    output(labelToIndex(labeledPoint.label)) = 0.9
    (labeledPoint.features, Vectors.dense(output))
  }

  protected def outputToLabel(output: Vector): Double = {
    val index = Bargmax(output.toBreeze.toDenseVector)
    indexToLabel(index)
  }
}

class ANN2ClassifierModel private[mllib](val annModel: FeedForwardModel,
                                        val labelToIndex: Map[Double, Int])
  extends ClassificationModel with ANN2ClassifierHelper with Serializable {
  /**
   * Predict values for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return predicted category from the trained model
   */
  override def predict(testData: Vector): Double = {
    val output = annModel.predict(testData)
    outputToLabel(output)
  }
}

class ANN2Classifier private(val labelToIndex: Map[Double, Int],
                            private val topology: Topology,
                            private val weights: Vector,
                            private val maxIterations: Int,
                            private val stepSize: Double,
                            private val convergeTol: Double)
  extends ANN2ClassifierHelper with Serializable {

  def run(data: RDD[LabeledPoint], batchSize: Int = 1): ANN2ClassifierModel = {
    val annData = data.map(lp => labeledPointToVectorPair(lp))
    /* train the model */
    val model = FeedForwardNetwork.train(annData, batchSize, maxIterations, topology, weights)
    new ANN2ClassifierModel(model, labelToIndex)
  }
}

/**
 * Top level methods for training the classifier based on artificial neural network (ANN)
 */
object ANN2Classifier {

  private val defaultStepSize = 1.0
  private val defaultBatchSize = 1

  /**
   * Trains an ANN classifier.
   *
   * @param data RDD containing labeled points for training.
   * @param batchSize batch size - number of instances to process in batch
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxIterations specifies maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS
   * @return ANN model.
   */
  def train(data: RDD[LabeledPoint],
            batchSize: Int,
            hiddenLayersTopology: Array[Int],
            maxIterations: Int,
            convergenceTol: Double): ANN2ClassifierModel = {
    val initialWeights = randomWeights(data, hiddenLayersTopology)
    train(data, batchSize, hiddenLayersTopology,
      initialWeights, maxIterations, defaultStepSize, convergenceTol)
  }

  /**
   * Trains an already pre-trained ANN classifier.
   * Assumes that the data has the same labels that the
   * data that were used for training, or at least the
   * subset of that labels
   *
   * @param data RDD containing labeled points for training.
   * @param batchSize batch size - number of instances to process in batch
   * @param model a pre-trained ANN classifier model.
   * @param maxIterations specifies maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS
   * @return ANN classifier model.
   */
  def train(data: RDD[LabeledPoint],
            batchSize: Int,
            model: ANN2ClassifierModel,
            maxIterations: Int,
            convergenceTol: Double): ANN2ClassifierModel = {
    assert(false, "Not supported yet.")
    null
  }

  /**
   * Trains an ANN classifier.
   *
   * @param data RDD containing labeled points for training.
   * @param batchSize batch size - number of instances to process in batch
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param initialWeights initial weights of underlying artificial neural network
   * @param maxIterations specifies maximum number of training iterations.
   * @param stepSize step size (not implemented)
   * @param convergenceTol convergence tolerance for LBFGS
   * @return ANN model.
   */
  def train(data: RDD[LabeledPoint],
            batchSize: Int,
            hiddenLayersTopology: Array[Int],
            initialWeights: Vector,
            maxIterations: Int,
            stepSize: Double,
            convergenceTol: Double): ANN2ClassifierModel = {
    val labelToIndex = data.map( lp => lp.label).distinct().collect().sorted.zipWithIndex.toMap
    val numIn = data.first().features.size
    val numOut = labelToIndex.size
    new ANN2Classifier(labelToIndex,
      Topology.multiLayerPerceptron(numIn +: hiddenLayersTopology :+ numOut),
      initialWeights, maxIterations, stepSize, convergenceTol).run(data, batchSize)
  }

  /**
   * Trains an ANN classifier.
   *
   * @param data RDD containing labeled points for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxIterations specifies maximum number of training iterations.
   * @param stepSize step size (not implemented)
   * @param convergenceTol convergence tolerance for LBFGS
   * @return ANN classifier model.
   */
  def train(data: RDD[LabeledPoint],
            hiddenLayersTopology: Array[Int],
            maxIterations: Int,
            stepSize: Double,
            convergenceTol: Double): ANN2ClassifierModel = {
    val initialWeights = randomWeights(data, hiddenLayersTopology)
    train(data, defaultBatchSize, hiddenLayersTopology, initialWeights, maxIterations, stepSize,
      convergenceTol)
  }

  /**
   * Trains an already pre-trained ANN classifier.
   * Assumes that the data has the same labels that the
   * data that were used for training, or at least the
   * subset of that labels
   *
   * @param data RDD containing labeled points for training.
   * @param model a pre-trained ANN classifier model.
   * @param maxIterations specifies maximum number of training iterations.
   * @param stepSize step size (not implemented)
   * @param convergenceTol convergence tolerance for LBFGS
   * @return ANN classifier model.
   */
  def train(data: RDD[LabeledPoint],
            model: ANN2ClassifierModel,
            maxIterations: Int,
            stepSize: Double,
            convergenceTol: Double): ANN2ClassifierModel = {
    assert(false, "Not supported yet.")
    null
  }

  /**
   * Trains an ANN classifier with one hidden layer of size (featureCount / 2 + 1)
   * with 2000 steps of size 1.0 and tolerance 1e-4
   *
   * @param data RDD containing labeled points for training.
   * @return ANN classifier model.
   */
  def train(data: RDD[LabeledPoint]): ANN2ClassifierModel = {
    val featureCount = data.first().features.size
    val hiddenSize = featureCount / 2 + 1
    val hiddenLayersTopology = Array[Int](hiddenSize)
    train(data, hiddenLayersTopology, 2000, 1.0, 1e-4)
  }

  /**
   * Returns random weights for the ANN classifier with the given hidden layers
   * and data dimensionality, i.e. the weights for the following topology:
   * [numFeatures -: hiddenLayers :- numLabels]
   *
   * @param data RDD containing labeled points for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed
   * @return vector with random weights.
   */
  def randomWeights(data: RDD[LabeledPoint],
                    hiddenLayersTopology: Array[Int], seed: Int): Vector = {
    /* TODO: remove duplication - the same analysis will be done in ANN2Classifier.run() */
    val labelCount = data.map( lp => lp.label).distinct().collect().length
    val featureCount = data.first().features.size
    val topology = Topology.multiLayerPerceptron(featureCount +: hiddenLayersTopology :+ labelCount)
    FeedForwardModel(topology, seed).weights()
  }

  /**
   * Returns random weights for the ANN classifier with the given hidden layers
   * and data dimensionality, i.e. the weights for the following topology:
   * [numFeatures -: hiddenLayers :- numLabels]
   *
   * @param data RDD containing labeled points for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @return vector with random weights.
   */
  def randomWeights(data: RDD[LabeledPoint], hiddenLayersTopology: Array[Int]): Vector = {
    randomWeights(data, hiddenLayersTopology, Random.nextInt())
  }
}
