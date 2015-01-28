package org.apache.spark.mllib.ann

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *, sum => Bsum}
import breeze.numerics.{sigmoid => Bsigmoid}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.Gradient

trait Layer {

  val weights: BDM[Double]
  val bias: BDV[Double]
  def activationInPlace(data: BDM[Double]): Unit
  def activationDerivative(output: BDM[Double]): BDM[Double]

  val size = weights.rows
  val inputSize = weights.cols

  def evaluate(data: BDM[Double]): BDM[Double] = {
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

class SigmoidLayer(val weights: BDM[Double], val bias: BDV[Double]) extends Layer {

  override def activationInPlace(data: BDM[Double]): Unit = Bsigmoid(data)

  override def activationDerivative(output: BDM[Double]): BDM[Double] = {
    val derivative = BDM.ones[Double](output.rows, output.cols)
    derivative :-= output
    derivative :*= output
    derivative
  }
}

class FeedForwardANN(val layers: Array[Layer]) {

  protected val weightCount =
    (for(i <- 0 until layers.length) yield
      (layers(i).size * layers(i).inputSize + layers(i).size)).sum

  def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val inputs = new Array[BDM[Double]](layers.length + 1)
    inputs(0) = data
    for(i <- 0 until layers.size) {
      inputs(i + 1) = layers(i).evaluate(inputs(i))
    }
    inputs
  }

  protected def unrollWeights(weights: linalg.Vector): (Array[BDM[Double]], Array[BDV[Double]]) = {
    require(weights.size == weightCount)
    val weightsCopy = weights.toArray
    val weightMatrices = new Array[BDM[Double]](layers.length)
    val bias = new Array[BDV[Double]](layers.length)
    var offset = 0
    for(i <- 0 until layers.length){
      weightMatrices(i) = new BDM[Double](layers(i).size, layers(i).inputSize, weightsCopy, offset)
      offset += layers(i).size * layers(i).inputSize
      bias(i) = (new BDV[Double](weightsCopy, offset, 1, layers(i).size))
      offset += layers(i).size
    }
    (weightMatrices, bias)
  }

  def rollWeights(weightMatricesUpdate: Array[BDM[Double]],
                            biasUpdate: Array[BDV[Double]],
                            cumGradient: Vector): Unit = {
    val wu = cumGradient.toArray
    var offset = 0
    for(i <- 0 until layers.length){
      var k = 0
      val numElements = layers(i).size * layers(i).inputSize
      while(k < numElements){
        wu(offset + k) += weightMatricesUpdate(i).data(k)
        k += 1
      }
      offset += numElements
      k = 0
      while(k < layers(i).size){
        wu(offset + k) += biasUpdate(i).data(k)
        k += 1
      }
      offset += layers(i).size
    }
  }
}

/* TODO: modelCreator function might grab other unrelated things in closure! */
private class BackPropagationGradient(val batchSize: Int,
                                      modelCreator: Vector => FeedForwardANN)
  extends Gradient {

   override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(data: Vector, label: Double, weights: Vector,
                       cumGradient: Vector): Double = {
    val model = modelCreator(weights)
    val layers = model.layers
    val arrData = data.toArray
    val inputSize = layers(0).inputSize
    val outputSize = layers.last.size
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
        layers(i).delta(outputs(i), deltas(i + 1), layers(i + 1).weights)
      }
      gradientMatrices(i) = deltas(i) * outputs(i - 1).t
      /* NB! dividing by the number of instances in
       * the batch to be transparent for the optimizer */
      gradientMatrices(i) :/= outputs(i).cols.toDouble
      avgDeltas(i) = Bsum(deltas(i)(*, ::))
      avgDeltas(i) :/= outputs(i).cols.toDouble
    }
    //(gradientMatrices, avgDeltas)
    model.rollWeights(gradientMatrices, avgDeltas, cumGradient)
    /* error */
    val diff = target :- outputs(layers.length)
    val outerError = Bsum(diff :* diff) / 2
    /* NB! dividing by the number of instances in
     * the batch to be transparent for the optimizer */
    outerError / realBatchSize
  }
}



