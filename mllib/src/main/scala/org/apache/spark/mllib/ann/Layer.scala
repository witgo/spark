package org.apache.spark.mllib.ann

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *, sum => Bsum}
import breeze.numerics.{sigmoid => Bsigmoid}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.{LBFGS, Gradient}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

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

  /* TODO: how to parametrize error? otherwise one has to override this */
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

class SigmoidLayer(val numIn: Int, val numOut: Int) extends Layer {

  override def activationInPlace(data: BDM[Double]): Unit = Bsigmoid(data)

  override def activationDerivative(output: BDM[Double]): BDM[Double] = {
    val derivative = BDM.ones[Double](output.rows, output.cols)
    derivative :-= output
    derivative :*= output
    derivative
  }
}

class FeedForwardANN(val layers: Array[Layer]) extends Serializable {

}

object FeedForwardANN {

  def multiLayerPerceptron(topology: Array[Int]): FeedForwardANN = {
    val layers = new Array[Layer](topology.length - 1)
    for(i <- 0 until topology.length - 1){
      layers(i) = new SigmoidLayer(topology(i), topology(i + 1))
    }
    new FeedForwardANN(layers)
  }

  def multiLayerPerceptron(data: RDD[(Vector, Vector)], hiddenLayersTopology: Array[Int]): FeedForwardANN = {
    val dataSample = data.first()
    val topology = dataSample._1.size +: hiddenLayersTopology :+ dataSample._2.size
    multiLayerPerceptron(topology)
  }
}

class FeedForwardANNModel(val config: FeedForwardANN, val weights: Array[BDM[Double]],
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

object FeedForwardANNModel {

  def apply(config: FeedForwardANN): FeedForwardANNModel = {
    val (weights, bias) = randomWeights(config)
    new FeedForwardANNModel(config, weights, bias)
  }

  def apply(config: FeedForwardANN, weightsAndBias: Vector): FeedForwardANNModel = {
    val (weights, bias) = unrollWeights(weightsAndBias, config.layers)
    new FeedForwardANNModel(config, weights, bias)
  }

  def randomWeights(config: FeedForwardANN, seed: Long = 11L): (Array[BDM[Double]], Array[BDV[Double]]) = {
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

  def randomWeights2(config: FeedForwardANN, seed: Long = 11L): Vector = {
    val (weights, bias) = randomWeights(config, seed)
    rollWeights(weights, bias)
  }


  protected def unrollWeights(weights: linalg.Vector,
                              layers: Array[Layer]): (Array[BDM[Double]], Array[BDV[Double]]) = {
    //require(weights.size == weightCount)
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
    val total = (for(i <- 0 until weightMatrices.size) yield (weightMatrices(i).size + bias(i).length)).sum
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

/* TODO: modelCreator function might grab other unrelated things in closure! */
private class BackPropagationGradient(val batchSize: Int,
                                      val config: FeedForwardANN)
  extends Gradient {

   override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(data: Vector, label: Double, weights: Vector,
                       cumGradient: Vector): Double = {
    val model = FeedForwardANNModel(config, weights)
    val layers = model.config.layers
    val arrData = data.toArray
    val inputSize = layers(0).numIn
    val outputSize = layers.last.numOut
    val realBatchSize = arrData.length / (inputSize + outputSize)
    val input = new BDM(inputSize, realBatchSize, arrData)
    val target = new BDM(outputSize, realBatchSize, arrData, inputSize * realBatchSize)

    /* forward run */

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
    //(gradientMatrices, avgDeltas)
    FeedForwardANNModel.rollWeights(gradientMatrices, avgDeltas, cumGradient)
    /* error */
    val diff = target :- outputs.last
    val outerError = Bsum(diff :* diff) / 2
    /* NB! dividing by the number of instances in
     * the batch to be transparent for the optimizer */
    outerError / realBatchSize
  }
}

class FFANN private[mllib](
                                              config: FeedForwardANN,
                                              maxNumIterations: Int,
                                              convergenceTol: Double,
                                              batchSize: Int = 1)
  extends Serializable {

  private val gradient = new BackPropagationGradient(batchSize, config)
  private val updater = new ANNUpdater()
  private val optimizer = new LBFGS(gradient, updater).
    setConvergenceTol(convergenceTol).
    setNumIterations(maxNumIterations)

  /**
   * Trains the ANN model.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights the initial weights of the ANN
   * @return ANN model.
   */
  private def run(trainingRDD: RDD[(Vector, Vector)],
                  initialWeights: Vector): FeedForwardANNModel = {
    val data = trainingRDD.map(v =>
        (0.0, Vectors.fromBreeze(
          BDV.vertcat(v._1.toBreeze.toDenseVector, v._2.toBreeze.toDenseVector))
          )
    )
    val weights = optimizer.optimize(data, initialWeights)
    FeedForwardANNModel(config, weights)
  }
}

object FFANN {

  def train(trainingRDD: RDD[(Vector, Vector)],
            batchSize: Int,
            maxIterations: Int,
            config: FeedForwardANN,
            initialWeights: Vector) = {
    new FFANN(config, maxIterations, 1e-4, 1).run(trainingRDD, initialWeights)
  }
}


