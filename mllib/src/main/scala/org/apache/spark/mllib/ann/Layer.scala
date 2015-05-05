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

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, axpy => brzAxpy, sum => Bsum}
import breeze.numerics.{log => Blog, sigmoid => Bsigmoid}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

/* Trait that holds Layer properties, that are needed to instantiate it.
*  Implements Layer instantiation.
* */
trait Layer extends Serializable {
  def getInstance(weights: Vector, position: Int): LayerModel
  def getInstance(seed: Long): LayerModel
}

/* Trait that holds Layer parameters aka weights.
*  Implements funtions needed for forward propagation, computing delta and gradient.
*  Can return weights in Vector format.
* */
trait LayerModel extends Serializable {
  val size: Int
  def eval(data: BDM[Double]): BDM[Double]
  def prevDelta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double]
  def grad(delta: BDM[Double], input: BDM[Double]): Vector
  def weights(): Vector
}

/* Layer for affine transformations that is y=Ax+b
* */
class AffineLayer(numIn: Int, numOut: Int) extends Layer {
  override def getInstance(weights: Vector, position: Int): LayerModel = {
    val (w, b) = AffineLayerModel.unroll(weights, position, numIn, numOut)
    AffineLayerModel(w, b)
  }

  override def getInstance(seed: Long = 11L): LayerModel = {
    val (w, b) = AffineLayerModel.randomWeights(numIn, numOut, seed)
    AffineLayerModel(w, b)
  }
}

/* Model of affine Layer
* */
class AffineLayerModel private(w: BDM[Double], b: BDV[Double]) extends LayerModel {
  val size = w.size + b.length

  // TODO: if the last batch != given batch ?
  private var z: BDM[Double] = null
  private var d: BDM[Double] = null
  private var gw: BDM[Double] = null
  private var gb: BDV[Double] = null
  private var ones: BDV[Double] = null

  override def eval(data: BDM[Double]): BDM[Double] = {
    if (z == null || z.cols != data.cols) z = new BDM[Double](w.rows, data.cols)
    z(::, *) := b
    BreezeUtil.dgemm(1.0, w, data, 1.0, z)
    z
  }

  override def prevDelta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double] = {
    if (d == null || d.cols != nextDelta.cols) d = new BDM[Double](w.cols, nextDelta.cols)
    BreezeUtil.dgemm(1.0, w.t, nextDelta, 0.0, d)
    d
  }

  override def grad(delta: BDM[Double], input: BDM[Double]): Vector = {
    if (gw == null || gw.cols != input.rows) gw = new BDM[Double](delta.rows, input.rows)
    BreezeUtil.dgemm(1.0 / input.cols, delta, input.t, 0.0, gw)
    if (gb == null || gb.length != delta.rows) gb = BDV.zeros[Double](delta.rows)
    if (ones == null || ones.length != delta.cols) ones = BDV.ones[Double](delta.cols)
    BreezeUtil.gemv(1.0 / input.cols, delta, ones, 0.0, gb)
    AffineLayerModel.roll(gw, gb)
  }

  override def weights(): Vector = AffineLayerModel.roll(w, b)
}

object AffineLayerModel {

  def apply(w: BDM[Double], b: BDV[Double]): AffineLayerModel = {
    new AffineLayerModel(w, b)
  }

  def unroll(weights: Vector, position: Int,
             numIn: Int, numOut: Int): (BDM[Double], BDV[Double]) = {
    val weightsCopy = weights.toArray
    // TODO: the array is not copied to BDMs, make sure this is OK!
    val w = new BDM[Double](numOut, numIn, weightsCopy, position)
    val b = new BDV[Double](weightsCopy, position + (numOut * numIn), 1, numOut)
    (w, b)
  }

  def roll(w: BDM[Double], b: BDV[Double]): Vector = {
    val result = new Array[Double](w.size + b.length)
    // TODO: make sure that we need to copy!
    System.arraycopy(w.toArray, 0, result, 0, w.size)
    System.arraycopy(b.toArray, 0, result, w.size, b.length)
    Vectors.dense(result)
  }

  def randomWeights(numIn: Int, numOut: Int, seed: Long = 11L): (BDM[Double], BDV[Double]) = {
    val rand: XORShiftRandom = new XORShiftRandom(seed)
    val weights = BDM.fill[Double](numOut, numIn){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    val bias = BDV.fill[Double](numOut){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    (weights, bias)
  }
}

/* Collection of functions and their derivatives for functional layers
* */
trait ActivationFunction extends Serializable {

  def eval(x: BDM[Double], y: BDM[Double]): Unit

  def derivative(x: BDM[Double], y: BDM[Double]): Unit

  def crossEntropy(target: BDM[Double], output: BDM[Double], result: BDM[Double]): Double

  def squared(target: BDM[Double], output: BDM[Double], result: BDM[Double]): Double
}

object ActivationFunction {

  def apply(x: BDM[Double], y: BDM[Double], func: Double => Double): Unit = {
    var i = 0
    while (i < x.rows) {
      var j = 0
      while (j < x.cols) {
        y(i, j) = func(x(i,j))
        j += 1
      }
      i += 1
    }
  }

  def apply(x1: BDM[Double], x2: BDM[Double], y: BDM[Double],
            func: (Double, Double) => Double): Unit = {
    var i = 0
    while (i < x1.rows) {
      var j = 0
      while (j < x1.cols) {
        y(i, j) = func(x1(i,j), x2(i, j))
        j += 1
      }
      i += 1
    }
  }

}

class SoftmaxFunction extends ActivationFunction {
  override def eval(x: BDM[Double], y: BDM[Double]): Unit = {
    var j = 0
    while (j < x.cols) {
      var i = 0
      var sum = 0.0
      while (i < x.rows) {
        val res = Math.exp(x(i,j))
        y(i, j) = res
        sum += res
        i += 1
      }
      i = 0
      while (i < x.rows) {
        y(i, j) /= sum
        i += 1
      }
      j += 1
    }
  }

  override def crossEntropy(output: BDM[Double], target: BDM[Double],
                            result: BDM[Double]): Double = {
    def m(o: Double, t: Double): Double = o - t
    ActivationFunction(output, target, result, m)
    -Bsum( target :* Blog(output)) / output.cols
  }

  override def derivative(x: BDM[Double], y: BDM[Double]): Unit = {
    def sd(z: Double): Double = (1 - z) * z
    ActivationFunction(x, y, sd)
  }

  override def squared(output: BDM[Double], target: BDM[Double], result: BDM[Double]): Double = {
    throw new UnsupportedOperationException("Sorry, squared error is not defined for SoftMax.")
  }
}

class SigmoidFunction extends ActivationFunction {
  override def eval(x: BDM[Double], y: BDM[Double]): Unit = {
    def s(z: Double): Double = Bsigmoid(z)
    ActivationFunction(x, y, s)
  }

  override def crossEntropy(output: BDM[Double], target: BDM[Double],
                            result: BDM[Double]): Double = {
    def m(o: Double, t: Double): Double = o - t
    ActivationFunction(output, target, result, m)
    -Bsum( target :* Blog(output)) / output.cols
  }

  override def derivative(x: BDM[Double], y: BDM[Double]): Unit = {
    def sd(z: Double): Double = (1 - z) * z
    ActivationFunction(x, y, sd)
  }

  override def squared(output: BDM[Double], target: BDM[Double], result: BDM[Double]): Double = {
    // TODO: make it readable
    def m(o: Double, t: Double): Double = (o - t)
    ActivationFunction(output, target, result, m)
    val e = Bsum(result :* result) / 2 / output.cols
    def m2(x: Double, o: Double) = x * (o - o * o)
    ActivationFunction(result, output, result, m2)
    e
  }
}


/* Functional layer, that is y = f(x)
* */
class FunctionalLayer (activationFunction: ActivationFunction) extends Layer {
  override def getInstance(weights: Vector, position: Int): LayerModel = getInstance(0L)

  override def getInstance(seed: Long): LayerModel =
    FunctionalLayerModel(activationFunction)
}

/* Functional layer model. Holds no parameters (weights).
* */
class FunctionalLayerModel private (val activationFunction: ActivationFunction
                                     ) extends LayerModel {
  val size = 0

  private var f: BDM[Double] = null
  private var d: BDM[Double] = null
  private var e: BDM[Double] = null

  override def eval(data: BDM[Double]): BDM[Double] =  {
    if (f == null || f.cols != data.cols) f = new BDM[Double](data.rows, data.cols)
    activationFunction.eval(data, f)
    f
  }

  override def prevDelta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double] = {
    if (d == null || d.cols != nextDelta.cols) d = new BDM[Double](nextDelta.rows, nextDelta.cols)
    activationFunction.derivative(input, d)
    d :*= nextDelta
    d
  }

  override def grad(delta: BDM[Double], input: BDM[Double]): Vector =
    Vectors.dense(new Array[Double](0))

  override def weights(): Vector = Vectors.dense(new Array[Double](0))

  def crossEntropy(output: BDM[Double], target: BDM[Double]): (BDM[Double], Double) = {
    if (e == null || e.cols != output.cols) e = new BDM[Double](output.rows, output.cols)
    val error = activationFunction.crossEntropy(output, target, e)
    (e, error)
  }

  def squared(output: BDM[Double], target: BDM[Double]): (BDM[Double], Double) = {
    if (e == null || e.cols != output.cols) e = new BDM[Double](output.rows, output.cols)
    val error = activationFunction.squared(output, target, e)
    (e, error)
  }

  def error(output: BDM[Double], target: BDM[Double]): (BDM[Double], Double) = {
    // TODO: allow user pick error
    activationFunction match {
      case sigmoid: SigmoidFunction => squared(output, target)
      case softmax: SoftmaxFunction => crossEntropy(output, target)
    }
  }
}

object FunctionalLayerModel {
  def apply(activationFunction: ActivationFunction): FunctionalLayerModel = {
    new FunctionalLayerModel(activationFunction)
  }
}

/* Network topology that holds the array of layers.
* */
class Topology(val layers: Array[Layer], val dropoutProb: Array[Double]) extends Serializable {

}

/* Factory for some of the frequently-used topologies
* */
object Topology {
  def apply(layers: Array[Layer]): Topology = {
    new Topology(layers, Array.fill[Double](layers.length)(0D))
  }

  def multiLayerPerceptron(layerSizes: Array[Int], softmax: Boolean = true): Topology = {
    val layers = new Array[Layer]((layerSizes.length - 1) * 2)
    for(i <- 0 until layerSizes.length - 1){
      layers(i * 2) = new AffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2 + 1) =
        if (softmax && i == layerSizes.length - 2) {
          new FunctionalLayer(new SoftmaxFunction())
        } else {
          new FunctionalLayer(new SigmoidFunction())
        }
    }
    Topology(layers)
  }
}

/* Model of Feed Forward Neural Network.
* Implements forward, gradient computation and can return weights in vector format.
* */
class FeedForwardModel(val layerModels: Array[LayerModel],
                       val topology: Topology) extends Serializable {
  def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](layerModels.length)
    outputs(0) = layerModels(0).eval(data)
    for(i <- 1 until layerModels.length){
      outputs(i) = layerModels(i).eval(outputs(i-1))
    }
    outputs
  }

  def computeGradient(data: BDM[Double], target: BDM[Double], cumGradient: Vector,
                      realBatchSize: Int): Double = {
    val outputs = forward(data)
    val deltas = new Array[BDM[Double]](layerModels.length)
    val L = layerModels.length - 1
    val (newE, newError) = layerModels.last match {
      case flm: FunctionalLayerModel => flm.error(outputs.last, target)
      case _ =>
        throw new UnsupportedOperationException("Non-functional layer not supported at the top")
    }
    deltas(L) = new BDM[Double](0, 0)
    deltas(L - 1) = newE
    for (i <- (L - 2) to (0, -1)) {
      deltas(i) = layerModels(i + 1).prevDelta(deltas(i + 1), outputs(i + 1))
    }
    val grads = new Array[Vector](layerModels.length)
    for (i <- 0 until layerModels.length) {
      val input = if (i==0) data else outputs(i - 1)
      grads(i) = layerModels(i).grad(deltas(i), input)
    }
    // update cumGradient
    val cumGradientArray = cumGradient.toArray
    var offset = 0
    // TODO: extract roll
    for (i <- 0 until grads.length) {
      val gradArray = grads(i).toArray
      var k = 0
      while (k < gradArray.length) {
        cumGradientArray(offset + k) += gradArray(k)
        k += 1
      }
      offset += gradArray.length
    }

    newError
  }

  def weights(): Vector = {
    // TODO: extract roll
    var size = 0
    for(i <- 0 until layerModels.length) {
      size += layerModels(i).size
    }
    val array = new Array[Double](size)
    var offset = 0
    for(i <- 0 until layerModels.length) {
      val layerWeights = layerModels(i).weights().toArray
      System.arraycopy(layerWeights, 0, array, offset, layerWeights.length)
      offset += layerWeights.length
    }
    Vectors.dense(array)
  }

  def predict(data: Vector): Vector = {
    val result = forward(data.toBreeze.toDenseVector.toDenseMatrix.t)
    Vectors.dense(result.last.toArray)
  }

}

object FeedForwardModel {
  def apply(topology: Topology, weights: Vector): FeedForwardModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    var offset = 0
    for(i <- 0 until layers.length){
      layerModels(i) = layers(i).getInstance(weights, offset)
      offset += layerModels(i).size
    }
    new FeedForwardModel(layerModels, topology)
  }

  def apply(topology: Topology, seed: Long = 11L): FeedForwardModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    var offset = 0
    for(i <- 0 until layers.length){
      layerModels(i) = layers(i).getInstance(seed)
      offset += layerModels(i).size
    }
    new FeedForwardModel(layerModels, topology)
  }
}

/* Neural network gradient. Does nothing but calling Model's gradient
* */
class ANNGradient(topology: Topology, dataStacker: DataStacker) extends Gradient {

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(data: Vector, label: Double, weights: Vector,
                       cumGradient: Vector): Double = {
    val (input, target, realBatchSize) = dataStacker.unstack(data)
    val model = FeedForwardModel(topology, weights)
    model.computeGradient(input, target, cumGradient, realBatchSize)
  }
}

/* Class that stacks the training samples in one vector allowing them to pass
*  through Optimizer/Gradient interfaces and thus allowing batch gradient computation.
*  Can unstack the training samples into matrices.
* */
class DataStacker(batchSize: Int, inputSize: Int, outputSize: Int) extends Serializable {
  def stack(data: RDD[(Vector, Vector)]): RDD[(Double, Vector)] = {
    val stackedData = if (batchSize == 1) {
      data.map(v =>
        (0.0,
          Vectors.fromBreeze(BDV.vertcat(
            v._1.toBreeze.toDenseVector,
            v._2.toBreeze.toDenseVector))
          ))
    } else {
      data.mapPartitions { it =>
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
    stackedData
  }

  def unstack(data: Vector): (BDM[Double], BDM[Double], Int) = {
    val arrData = data.toArray
    val realBatchSize = arrData.length / (inputSize + outputSize)
    val input = new BDM(inputSize, realBatchSize, arrData)
    val target = new BDM(outputSize, realBatchSize, arrData, inputSize * realBatchSize)
    (input, target, realBatchSize)
  }
}

/* Simple updater
* */
private class ANNUpdater extends Updater {

  override def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       iter: Int,
                       regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}
/* MLlib-style trainer class that trains a network given the data and topology
* */
class FeedForwardTrainer (topology: Topology, val inputSize: Int,
                          val outputSize: Int) extends Serializable {

  // TODO: what if we need to pass random seed?
  private var _weights = FeedForwardModel(topology).weights()
  private var _batchSize = 1
  private var dataStacker = new DataStacker(_batchSize, inputSize, outputSize)
  private var _gradient: Gradient = new ANNGradient(topology, dataStacker)
  private var _updater: Updater = new ANNUpdater()
  private var optimizer: Optimizer = LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(100)


  def getWeights: Vector = _weights

  def setWeights(value: Vector): FeedForwardTrainer = {
    _weights = value
    this
  }

  def setBatchSize(value: Int): FeedForwardTrainer = {
    _batchSize = value
    dataStacker = new DataStacker(value, inputSize, outputSize)
    this
  }

  def SGDOptimizer: GradientDescent = {
    val sgd = new GradientDescent(_gradient, _updater)
    optimizer = sgd
    sgd
  }

  def LBFGSOptimizer: LBFGS = {
    val lbfgs = new LBFGS(_gradient, _updater)
    optimizer = lbfgs
    lbfgs
  }

  def setUpdater(value: Updater): FeedForwardTrainer =  {
    _updater = value
    updateUpdater(value)
    this
  }

  def setGradient(value: Gradient): FeedForwardTrainer = {
    _gradient = value
    updateGradient(value)
    this
  }

  private[this] def updateGradient(gradient: Gradient): Unit = {
    optimizer match {
      case lbfgs: LBFGS => lbfgs.setGradient(gradient)
      case sgd: GradientDescent => sgd.setGradient(gradient)
      case other => throw new UnsupportedOperationException(
        s"Only LBFGS and GradientDescent are supported but got ${other.getClass}.")
    }
  }

  private[this] def updateUpdater(updater: Updater): Unit = {
    optimizer match {
      case lbfgs: LBFGS => lbfgs.setUpdater(updater)
      case sgd: GradientDescent => sgd.setUpdater(updater)
      case other => throw new UnsupportedOperationException(
        s"Only LBFGS and GradientDescent are supported but got ${other.getClass}.")
    }
  }

  def train(data: RDD[(Vector, Vector)]): FeedForwardModel = {
    val newWeights = optimizer.optimize(dataStacker.stack(data), getWeights)
    FeedForwardModel(topology, newWeights)
  }

}
