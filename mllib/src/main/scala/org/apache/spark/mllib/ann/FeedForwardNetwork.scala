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

package org.apache.spark.mllib.ann

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *, sum => Bsum}
import breeze.numerics.{sigmoid => Bsigmoid}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.{LBFGS, Gradient}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

/* Layer does not contain weights to let it be a part of Topology, which is lightweight object
 * that goes in closure with Gradient. New types of layers are implemented by overriding
 * activation function and its derivative
 *
 * */
trait Layer extends Serializable {

  def activationInPlace(data: BDM[Double]): Unit
  def activationDerivative(output: BDM[Double]): BDM[Double]

  val numIn: Int
  val numOut: Int

  def evaluate(data: BDM[Double], weights: BDM[Double], bias: BDV[Double]): BDM[Double] = {
    val output = weights * data
    output(::, *) :+= bias
    activationInPlace(output)
    output
  }

  /* TODO: how to parametrize error & regularization ? otherwise one has to override these two */
  def delta(output: BDM[Double], target: BDM[Double]): BDM[Double] = {
    (output - target) :* activationDerivative(output)
  }

  def delta(output: BDM[Double], nextDelta: BDM[Double], nextWeights: BDM[Double]): BDM[Double] = {
    (nextWeights.t * nextDelta) :* activationDerivative(output)
  }
}

object Layer {

  def randomWeights(numIn: Int, numOut: Int, seed: Long = 11L): (BDM[Double], BDV[Double]) = {
    val rand: XORShiftRandom = new XORShiftRandom(seed)
    val weights = BDM.fill[Double](numOut, numIn){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    val bias = BDV.fill[Double](numOut){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    (weights, bias)
  }

}

/* Standard type of ANN layer */
class SigmoidLayer(val numIn: Int, val numOut: Int) extends Layer {

  override def activationInPlace(data: BDM[Double]): Unit = Bsigmoid.inPlace(data)

  override def activationDerivative(output: BDM[Double]): BDM[Double] = {
    val derivative = BDM.ones[Double](output.rows, output.cols)
    derivative :-= output
    derivative :*= output
    derivative
  }
}

/* Class that holds ANN topology, from which a network can be created
*
* */
class FeedForwardTopology(val layers: Array[Layer]) extends Serializable {

}

/* Fabric for standard topologies, such as multi-layer perceprtion, softmax etc.
*  User implements a custom topology in the same way.
* */
object FeedForwardTopology {

  def multiLayerPerceptron(topology: Array[Int]): FeedForwardTopology = {
    val layers = new Array[Layer](topology.length - 1)
    for(i <- 0 until topology.length - 1){
      layers(i) = new SigmoidLayer(topology(i), topology(i + 1))
    }
    new FeedForwardTopology(layers)
  }

  def multiLayerPerceptron(data: RDD[(Vector, Vector)],
                           hiddenLayersTopology: Array[Int]): FeedForwardTopology = {
    val dataSample = data.first()
    val topology = dataSample._1.size +: hiddenLayersTopology :+ dataSample._2.size
    multiLayerPerceptron(topology)
  }
}

/* Model for feed forward network. Holds weights.
 * Can be instantiated with a topology and weights.
 * */
class FeedForwardModel(val config: FeedForwardTopology, val weights: Array[BDM[Double]],
                          val bias: Array[BDV[Double]]) extends Serializable {

  protected val layers = config.layers

  protected val weightCount =
    (for(i <- 0 until layers.length) yield
      (layers(i).numOut * layers(i).numIn + layers(i).numOut)).sum

  def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](layers.length)
    outputs(0) = layers(0).evaluate(data, weights(0), bias(0))
    for(i <- 1 until layers.size) {
      outputs(i) = layers(i).evaluate(outputs(i - 1), weights(i), bias(i))
    }
    outputs
  }

  def predict(data: Vector): Vector = {
    val result = forward(data.toBreeze.toDenseVector.toDenseMatrix.t)
    Vectors.dense(result.last.toArray)
  }

  def predict(data: BDM[Double]): BDM[Double] = {
    val result = forward(data)
    result.last
  }
}

/* Fabric for feed forward networks.
 * */
object FeedForwardModel {

  def apply(config: FeedForwardTopology): FeedForwardModel = {
    val (weights, bias) = randomWeights(config)
    new FeedForwardModel(config, weights, bias)
  }

  def apply(config: FeedForwardTopology, weightsAndBias: Vector): FeedForwardModel = {
    val (weights, bias) = unrollWeights(weightsAndBias, config.layers)
    new FeedForwardModel(config, weights, bias)
  }

  def randomWeights(config: FeedForwardTopology,
                    seed: Long = 11L): (Array[BDM[Double]], Array[BDV[Double]]) = {
    val numLayers = config.layers.length
    val weights = new Array[BDM[Double]](numLayers)
    val bias = new Array[BDV[Double]](numLayers)
    for(i <- 0 until numLayers){
      val (w, b) = Layer.randomWeights(config.layers(i).numIn, config.layers(i).numOut, seed)
      weights(i) = w
      bias(i) = b
    }
    (weights, bias)
  }

  def randomWeights2(config: FeedForwardTopology, seed: Long = 11L): Vector = {
    val (weights, bias) = randomWeights(config, seed)
    rollWeights(weights, bias)
  }


  protected def unrollWeights(weights: linalg.Vector,
                              layers: Array[Layer]): (Array[BDM[Double]], Array[BDV[Double]]) = {
    val weightsCopy = weights.toArray
    val weightMatrices = new Array[BDM[Double]](layers.length)
    val bias = new Array[BDV[Double]](layers.length)
    var offset = 0
    for(i <- 0 until layers.length){
      weightMatrices(i) = new BDM[Double](layers(i).numOut, layers(i).numIn, weightsCopy, offset)
      offset += layers(i).numOut * layers(i).numIn
      bias(i) = new BDV[Double](weightsCopy, offset, 1, layers(i).numOut)
      offset += layers(i).numOut
    }
    (weightMatrices, bias)
  }

  def rollWeights(weightMatrices: Array[BDM[Double]],
                  bias: Array[BDV[Double]]): Vector = {
    val total = (for(i <- 0 until weightMatrices.size) yield
      (weightMatrices(i).size + bias(i).length)).sum
    val flat = Vectors.dense(new Array[Double](total))
    rollWeights(weightMatrices, bias, flat)
    flat
  }

  def rollWeights(weightMatricesUpdate: Array[BDM[Double]],
                  biasUpdate: Array[BDV[Double]],
                  cumGradient: Vector): Unit = {
    val wu = cumGradient.toArray
    var offset = 0
    for(i <- 0 until weightMatricesUpdate.length){
      var k = 0
      val numElements = weightMatricesUpdate(i).size
      while(k < numElements){
        wu(offset + k) += weightMatricesUpdate(i).data(k)
        k += 1
      }
      offset += numElements
      k = 0
      while(k < biasUpdate(i).size){
        wu(offset + k) += biasUpdate(i).data(k)
        k += 1
      }
      offset += biasUpdate(i).size
    }
  }

}

/* Back propagation gradient. Implements feed forward network back propagation
* with the use of Layer's forward and delta functions that can be overridden
* to change the cost function
* */
private class BackPropagationGradient(val batchSize: Int,
                                      val config: FeedForwardTopology)
  extends Gradient {

   override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(data: Vector, label: Double, weights: Vector,
                       cumGradient: Vector): Double = {
    val model = FeedForwardModel(config, weights)
    val layers = model.config.layers
    val arrData = data.toArray
    val inputSize = layers(0).numIn
    val outputSize = layers.last.numOut
    val realBatchSize = arrData.length / (inputSize + outputSize)
    val input = new BDM(inputSize, realBatchSize, arrData)
    val target = new BDM(outputSize, realBatchSize, arrData, inputSize * realBatchSize)

    val outputs = model.forward(input)
    val deltas = new Array[BDM[Double]](layers.length)
    val gradientMatrices = new Array[BDM[Double]](layers.length)
    val avgDeltas = new Array[BDV[Double]](layers.length)
    /* back propagation */
    for(i <- (layers.length - 1) to (0, -1)){ /* until */
      deltas(i) = if (i == layers.length - 1) {
        layers(i).delta(outputs(i), target)
      } else {
        layers(i).delta(outputs(i), deltas(i + 1), model.weights(i + 1))
      }
      gradientMatrices(i) = if ( i == 0) {
        deltas(i) * input.t
      } else {
        deltas(i) * outputs(i - 1).t
      }
      /* NB! dividing by the number of instances in
       * the batch to be transparent for the optimizer */
      gradientMatrices(i) :/= outputs(i).cols.toDouble
      avgDeltas(i) = Bsum(deltas(i)(*, ::))
      avgDeltas(i) :/= outputs(i).cols.toDouble
    }
    FeedForwardModel.rollWeights(gradientMatrices, avgDeltas, cumGradient)
    /* error */
    val diff = target :- outputs.last
    val outerError = Bsum(diff :* diff) / 2
    /* NB! dividing by the number of instances in
     * the batch to be transparent for the optimizer */
    outerError / realBatchSize
  }
}

/* MLlib-style trainer class that trains a network given the data and topology
* */
class FeedForwardNetwork private[mllib](config: FeedForwardTopology, maxNumIterations: Int,
                                        convergenceTol: Double, batchSize: Int = 1)
  extends Serializable {

  private val gradient = new BackPropagationGradient(batchSize, config)
  private val updater = new ANNUpdater()
  private val optimizer = new LBFGS(gradient, updater).
    setConvergenceTol(convergenceTol).setNumIterations(maxNumIterations)

  /**
   * Trains the ANN model.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights the initial weights of the ANN
   * @return ANN model.
   */
  private def run(trainingRDD: RDD[(Vector, Vector)],
                  initialWeights: Vector): FeedForwardModel = {
    val inputSize = config.layers(0).numIn
    val outputSize = config.layers.last.numOut
    val data = if (batchSize == 1) {
      trainingRDD.map(v =>
        (0.0,
          Vectors.fromBreeze(BDV.vertcat(
            v._1.toBreeze.toDenseVector,
            v._2.toBreeze.toDenseVector))
          ))
    } else {
      trainingRDD.mapPartitions { it =>
        it.grouped(batchSize).map { seq =>
          val size = seq.size
          val bigVector = new Array[Double](inputSize * size + outputSize * size)
          var i = 0
          seq.foreach { case (in, out) =>
            System.arraycopy(in.toArray, 0, bigVector, i * inputSize, inputSize)
            System.arraycopy(out.toArray, 0, bigVector,
              inputSize * size + i * outputSize, outputSize)
            i += 1
          }
          (0.0, Vectors.dense(bigVector))
        }
      }
    }
    val weights = optimizer.optimize(data, initialWeights)
    FeedForwardModel(config, weights)
  }
}

/* MLlib-style object for the collection of train methods
 *
 */
object FeedForwardNetwork {

  def train(trainingRDD: RDD[(Vector, Vector)],
            batchSize: Int,
            maxIterations: Int,
            config: FeedForwardTopology,
            initialWeights: Vector) = {
    new FeedForwardNetwork(config, maxIterations, 1e-4, 1).run(trainingRDD, initialWeights)
  }
}


