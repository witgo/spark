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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Axis => BrzAxis, sum => brzSum}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

class RBMLayerModel(
  val w: BDM[Double],
  val vb: BDV[Double],
  val hb: BDV[Double],
  val dropoutRate: Double) extends TopologyModel {
  protected lazy val rand: XORShiftRandom = new XORShiftRandom(Utils.random.nextLong())
  val size = w.size + vb.length + hb.length
  lazy val visibleLayer: LayerModel = {
    new AffineLayerModel(w.t, vb)
  }
  lazy val hiddenLayer: LayerModel = {
    new AffineLayerModel(w, hb)
  }
  lazy val visibleActivateLayer: LayerModel = {
    new FunctionalLayerModel(new SigmoidFunction())
  }
  lazy val hiddenActivateLayer: LayerModel = {
    new FunctionalLayerModel(new SigmoidFunction())
  }
  val cdK: Int = 1

  def numOut: Int = w.rows

  def numIn: Int = w.cols

  protected def activateHidden(data: BDM[Double]): BDM[Double] = {
    require(data.rows == w.cols)
    hiddenActivateLayer.eval(hiddenLayer.eval(data))
  }

  protected def activateVisible(data: BDM[Double]): BDM[Double] = {
    require(data.rows == w.rows)
    visibleActivateLayer.eval(visibleLayer.eval(data))
  }

  protected def sampleHidden(hiddenMean: BDM[Double]): BDM[Double] = {
    hiddenMean.map(v => if (rand.nextDouble() < v) 1D else 0D)
  }

  protected def sampleVisible(visibleMean: BDM[Double]): BDM[Double] = {
    visibleMean.map(v => if (rand.nextDouble() < v) 1D else 0D)
  }

  protected def meanSquaredError(out: BDM[Double], label: BDM[Double]): Double = {
    require(label.rows == out.rows)
    require(label.cols == out.cols)
    var diff = 0D
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        diff += math.pow(label(i, j) - out(i, j), 2)
      }
    }
    diff / out.rows
  }

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  override def weights(): Vector = RBMLayerModel.roll(w, vb, hb)

  override def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val hidden = activateHidden(data)
    if (dropoutRate > 0) hidden :*= (1 - dropoutRate)
    Array(hidden)
  }

  protected def dropOutMask(cols: Int): BDM[Double] = {
    val mask = BDM.zeros[Double](numOut, cols)
    for (i <- 0 until numOut) {
      for (j <- 0 until cols) {
        mask(i, j) = if (rand.nextDouble() > dropoutRate) 1D else 0D
      }
    }
    mask
  }

  def computeGradient(
    input: BDM[Double],
    target: BDM[Double],
    cumGradient: Vector,
    realBatchSize: Int): Double = {

    val batchSize = input.cols
    val mask: BDM[Double] = if (dropoutRate > 0) {
      this.dropOutMask(batchSize)
    } else {
      null
    }

    val h1Mean = activateHidden(input)
    val h1Sample = sampleHidden(h1Mean)

    var vKMean: BDM[Double] = null
    var vKSample: BDM[Double] = null
    var hKMean: BDM[Double] = null
    var hKSample: BDM[Double] = h1Sample
    if (dropoutRate > 0) {
      hKSample :*= mask
    }

    for (i <- 0 until cdK) {
      vKMean = activateVisible(hKSample)
      hKMean = activateHidden(vKMean)
      hKSample = sampleHidden(hKMean)
      if (dropoutRate > 0) {
        hKSample :*= mask
      }
    }

    val gW: BDM[Double] = hKMean * vKMean.t - h1Mean * input.t
    val gVb = brzSum(vKMean - input, BrzAxis._1)
    val gHb = brzSum(hKMean - h1Mean, BrzAxis._1)

    val (cumW, cumVb, cumHb) = RBMLayerModel.unroll(cumGradient, 0, numIn, numOut)
    cumW :+= gW
    cumVb :+= gVb
    cumHb :+= gHb
    val mse = meanSquaredError(input, vKMean)
    mse
  }

  override def predict(data: Vector): Vector = {
    // throw new UnsupportedOperationException()
    val result = forward(data.toBreeze.toDenseVector.toDenseMatrix.t)
    Vectors.dense(result.last.toArray)
  }

}

object RBMLayerModel {
  def apply(
    numIn: Int,
    numOut: Int,
    dropoutRate: Double,
    weights: Vector,
    position: Int): RBMLayerModel = {
    val (w, vb, hb) = unroll(weights, position, numIn, numOut)
    new RBMLayerModel(w, vb, hb, dropoutRate)
  }

  def apply(numIn: Int, numOut: Int, dropoutRate: Double, seed: Long): RBMLayerModel = {
    val (w, vb, hb) = randomWeights(numIn, numOut, seed)
    new RBMLayerModel(w, vb, hb, dropoutRate)
  }

  def unroll(
    weights: Vector,
    position: Int,
    numIn: Int,
    numOut: Int): (BDM[Double], BDV[Double], BDV[Double]) = {
    val weightsCopy = weights.toArray
    var offset = position
    val w = new BDM[Double](numOut, numIn, weightsCopy, offset)
    offset += numOut * numIn
    val vb = new BDV[Double](weightsCopy, offset, 1, numIn)
    offset += numIn
    val hb = new BDV[Double](weightsCopy, offset, 1, numOut)
    offset += numOut
    (w, vb, hb)
  }

  def roll(w: BDM[Double], vb: BDV[Double], hb: BDV[Double]): Vector = {
    val result = new Array[Double](w.size + vb.length + hb.length)
    var offset = 0
    System.arraycopy(w.toArray, 0, result, offset, w.size)
    offset += w.size
    System.arraycopy(vb.toArray, 0, result, offset, vb.length)
    offset += vb.length
    System.arraycopy(hb.toArray, 0, result, offset, hb.length)
    offset += hb.length
    Vectors.dense(result)
  }

  def randomWeights(
    numIn: Int,
    numOut: Int,
    seed: Long = 11L): (BDM[Double], BDV[Double], BDV[Double]) = {
    val rand: XORShiftRandom = new XORShiftRandom(seed)
    val w = BDM.fill[Double](numOut, numIn) {
      rand.nextDouble * 4D * math.sqrt(6D / (numIn + numOut))
    }
    val vb = BDV.zeros[Double](numIn)
    val hb = BDV.zeros[Double](numOut)
    (w, vb, hb)
  }
}

class RBMTopology(
  val numIn: Int,
  val numOut: Int,
  val dropoutRate: Double) extends Topology {
  override def getInstance(weights: Vector): TopologyModel = {
    RBMLayerModel(numIn, numOut, dropoutRate, weights, 0)
  }

  override def getInstance(seed: Long): TopologyModel = {
    RBMLayerModel(numIn, numOut, dropoutRate, seed)
  }

  def getRBMLayerModel(weights: Vector, position: Int): RBMLayerModel = {
    RBMLayerModel(numIn, numOut, dropoutRate, weights, position)
  }

  def getRBMLayerModel(seed: Long): RBMLayerModel = {
    RBMLayerModel(numIn, numOut, dropoutRate, seed)
  }
}

object RBMTopology {
  def apply(numIn: Int, numOut: Int, dropoutRate: Double): RBMTopology = {
    new RBMTopology(numIn, numOut, dropoutRate)
  }
}

class RBMTrainer(topology: RBMTopology) extends Serializable {
  private val outputSize = 0
  private var _seed: Long = 41
  private var _weights = RBMLayerModel(topology.numIn, topology.numOut,
    topology.dropoutRate, _seed).weights()
  private var _batchSize = 1
  private var dataStacker = new DataStacker(_batchSize, topology.numIn, outputSize)
  private var _gradient: Gradient = new ANNGradient(topology, dataStacker)
  private var _updater: Updater = new ANNUpdater()
  private var optimizer: Optimizer = SGDOptimizer.setNumIterations(100)

  def getWeights: Vector = _weights

  def setWeights(value: Vector): this.type = {
    _weights = value
    this
  }

  def setBatchSize(value: Int): this.type = {
    _batchSize = value
    dataStacker = new DataStacker(value, topology.numIn, outputSize)
    this
  }

  def setSeed(value: Long): this.type = {
    _seed = value
    this
  }

  def SGDOptimizer: GradientDescent = {
    val sgd = new GradientDescent(_gradient, _updater)
    optimizer = sgd
    sgd
  }


  def setUpdater(value: Updater): this.type = {
    _updater = value
    updateUpdater(value)
    this
  }

  def setGradient(value: Gradient): this.type = {
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

  def train(data: RDD[(Vector)]): RBMLayerModel = {
    val newData = data.map(t => (t, Vectors.dense(Array.empty[Double])))
    val newWeights = optimizer.optimize(dataStacker.stack(newData), getWeights)
    topology.getInstance(newWeights).asInstanceOf[RBMLayerModel]
  }
}

class StackedRBMTrainer(topology: StackedRBMTopology) extends Serializable with Logging {

  private var seed: Long = 41
  private var batchSize = 1
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0

  def setBatchSize(value: Int): this.type = {
    batchSize = value
    this
  }

  def setSeed(value: Long): this.type = {
    seed = value
    this
  }

  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }


  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  def train(data: RDD[Vector], maxLayer: Int = -1): StackedRBMModel = {
    val trainLayer = if (maxLayer > -1D) maxLayer else topology.numLayer
    val stackedRBM = topology.getInstance(117).asInstanceOf[StackedRBMModel]
    for (layer <- 0 until trainLayer) {
      logInfo(s"Train ($layer/$trainLayer)")
      val dataBatch = forward(data, stackedRBM, layer)
      val rbmTopology = topology.innerRBMs(layer)
      val initialWeights = rbmTopology.getInstance(seed).weights()
      val trainer = new RBMTrainer(rbmTopology)
      trainer.SGDOptimizer.
        setNumIterations(numIterations).
        setMiniBatchFraction(miniBatchFraction).
        setStepSize(stepSize).
        setRegParam(regParam)
      // val updater = new AdaGradUpdater(0, 1e-6, 0.9)
      val updater = new EquilibratedUpdater(1e-6, 0)
      trainer.setWeights(initialWeights).setUpdater(updater).setBatchSize(batchSize)
      val rbmModel = trainer.train(dataBatch)
      stackedRBM.layerModels(layer) = rbmModel
    }
    stackedRBM
  }

  private def forward(
    data: RDD[Vector],
    stackedRBM: StackedRBMModel,
    toLayer: Int): RDD[Vector] = {
    if (toLayer > 0) {
      data.map(x => stackedRBM.predict(x, toLayer))
    } else {
      data
    }
  }

}

class StackedRBMTopology(val innerRBMs: Array[RBMTopology]) extends Logging with Topology {

  def numLayer: Int = innerRBMs.length

  def numInput: Int = innerRBMs.head.numIn

  def numOut: Int = innerRBMs.last.numOut

  def layers: Array[RBMTopology] = innerRBMs

  override def getInstance(weights: Vector): TopologyModel = {
    StackedRBMModel(this, weights)
  }

  override def getInstance(seed: Long): TopologyModel = {
    StackedRBMModel(this, seed)
  }
}

object StackedRBMTopology {
  def multiLayer(layerSizes: Array[Int]): StackedRBMTopology = {
    val numLayer = layerSizes.length - 1
    val innerRBMs: Array[RBMTopology] = new Array[RBMTopology](numLayer)
    for (layer <- 0 until numLayer) {
      val dropout = if (layer == numLayer - 1) {
        // 0.5
        0.0
      } else {
        0.0
      }
      innerRBMs(layer) = new RBMTopology(layerSizes(layer), layerSizes(layer + 1), dropout)
      println(s"innerRBMs($layer) = ${innerRBMs(layer).numIn} * ${innerRBMs(layer).numOut}")
    }
    new StackedRBMTopology(innerRBMs)
  }
}

class StackedRBMModel(
  val layerModels: Array[RBMLayerModel],
  val topology: StackedRBMTopology) extends TopologyModel {
  override def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](layerModels.length)
    outputs(0) = layerModels(0).forward(data).head
    for (i <- 1 until layerModels.length) {
      outputs(i) = layerModels(i).forward(outputs(i - 1)).head
    }
    outputs
  }

  def forward(data: BDM[Double], toLayer: Int): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](toLayer)
    outputs(0) = layerModels(0).forward(data).head
    for (i <- 1 until toLayer) {
      outputs(i) = layerModels(i).forward(outputs(i - 1)).head
    }
    outputs
  }

  override def weights(): Vector = {
    var size = 0
    layerModels.indices.foreach { i =>
      size += layerModels(i).size
    }
    val array = new Array[Double](size)
    var offset = 0
    layerModels.indices.foreach { i =>
      val layerWeights = layerModels(i).weights().toArray
      System.arraycopy(layerWeights, 0, array, offset, layerWeights.length)
      offset += layerWeights.length
    }
    Vectors.dense(array)
  }

  override def predict(data: Vector): Vector = {
    val result = forward(data.toBreeze.toDenseVector.toDenseMatrix.t)
    Vectors.dense(result.last.toArray)
  }

  def predict(data: Vector, toLayer: Int): Vector = {
    val result = forward(data.toBreeze.toDenseVector.toDenseMatrix.t, toLayer)
    Vectors.dense(result.last.toArray)
  }

  override def computeGradient(
    data: BDM[Double],
    target: BDM[Double],
    cumGradient: Vector,
    realBatchSize: Int): Double = {
    throw new UnsupportedOperationException()
  }

}

object StackedRBMModel {
  def apply(topology: StackedRBMTopology, weights: Vector): StackedRBMModel = {
    val layers = topology.layers
    val layerModels = new Array[RBMLayerModel](layers.length)
    var offset = 0
    layers.indices.foreach { i =>
      layerModels(i) = layers(i).getRBMLayerModel(weights, offset)
      offset += layerModels(i).size
    }
    new StackedRBMModel(layerModels, topology)
  }

  def apply(topology: StackedRBMTopology, seed: Long = 11L): StackedRBMModel = {
    val layers = topology.layers
    val layerModels = new Array[RBMLayerModel](layers.length)
    var offset = 0
    layers.indices.foreach { i =>
      layerModels(i) = layers(i).getRBMLayerModel(seed)
      offset += layerModels(i).size
    }
    new StackedRBMModel(layerModels, topology)
  }
}

class DBNTrainer(
  val stackedRBM: StackedRBMTopology,
  val topLayer: FeedForwardTopology) extends Logging {
  private var batchSize = 1
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private val rand = new XORShiftRandom(System.nanoTime())

  def setBatchSize(value: Int): this.type = {
    batchSize = value
    this
  }

  def setSeed(value: Long): this.type = {
    rand.setSeed(value)
    this
  }

  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }


  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  def toMLP(): DropoutTopology = {
    val numLayer = stackedRBM.numLayer * 2 + topLayer.layers.length
    val layers: Array[Layer] = new Array[Layer](numLayer)
    for (i <- 0 until stackedRBM.numLayer) {
      layers(i * 2) = new AffineLayer(stackedRBM.layers(i).numIn, stackedRBM.layers(i).numOut)
      layers(i * 2 + 1) = new FunctionalLayer(new SigmoidFunction())
    }

    topLayer.layers.indices.foreach { i =>
      layers(i + stackedRBM.numLayer * 2) = topLayer.layers(i)
    }
    new DropoutTopology(layers, 0, 0.5)
  }

  def pretrain(
    data: RDD[(Vector, Vector)]): StackedRBMModel = {
    val trainer = new StackedRBMTrainer(stackedRBM).
      setNumIterations(numIterations).
      setMiniBatchFraction(miniBatchFraction).
      setStepSize(stepSize).
      setRegParam(regParam).
      setBatchSize(batchSize)
    trainer.train(data.map(_._1))
  }

  def finetune(
    data: RDD[(Vector, Vector)],
    stackedRBMModel: StackedRBMModel = null): TopologyModel = {
    val s = data.take(1).head
    val inputSize: Int = s._1.size
    val outputSize: Int = s._2.size
    val topology = toMLP()
    val mlp = topology.getInstance(rand.nextLong()).asInstanceOf[DropoutModel]

    if (stackedRBMModel != null) {
      stackedRBMModel.layerModels.indices.foreach { i =>
        mlp.layerModels(i) match {
          case affineLayerModel: AffineLayerModel =>
            affineLayerModel.w := stackedRBMModel.layerModels(i).w
            affineLayerModel.b := stackedRBMModel.layerModels(i).hb
          case _ =>
        }
      }
    }

    val initialWeights = mlp.weights()
    val trainer = new FeedForwardTrainer(topology, inputSize, outputSize)
    trainer.setWeights(initialWeights)
    trainer.SGDOptimizer.
      setNumIterations(numIterations).
      setMiniBatchFraction(miniBatchFraction).
      setStepSize(stepSize).
      setRegParam(regParam)
    val updater = new EquilibratedUpdater(1e-6, 0.5)
    trainer.
      setWeights(initialWeights).
      setUpdater(updater).
      setBatchSize(batchSize)
    val model = trainer.train(data)
    model
  }
}

class MLPEquilibratedUpdater(
  val topology: DropoutTopology,
  epsilon: Double,
  momentum: Double) extends
EquilibratedUpdater(epsilon, momentum) {

  override protected def l2(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double): Double = {
    if (regParam == 0.0) return 0
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val w = topology.getInstance(weightsOld).asInstanceOf[DropoutModel]
    w.layerModels.indices.foreach { i =>
      w.layerModels(i) match {
        case affineLayerModel: AffineLayerModel =>
          affineLayerModel.w :*= (1.0 - thisIterStepSize * regParam)
        case _ =>
      }
    }
    0.0
  }
}
