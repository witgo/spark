package org.apache.spark.mllib.ann

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *}
import breeze.numerics.{sigmoid => Bsigmoid}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vector

trait Layer {

  val weights: BDM[Double]
  val bias: BDV[Double]
  def activationInPlace(data: BDM[Double]): Unit
  def activationDerivative(output: BDM[Double]): BDM[Double]

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

trait ArtNNHelper {

  val layers: Array[Layer]

  def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outArray = new Array[BDM[Double]](layers.length)
    outArray(0) = data
    for(i <- 1 until layers.size) {
      outArray(i) = layers(i).evaluate(outArray(i - 1))
    }
    outArray
  }

  def computeGradient(data: BDM[Double]): (Array[BDM[Double]], Array[BDV[Double]]) = {
    val outputs = forward(data)
    val deltas = new Array[BDM[Double]](layers.length)
    val gradientMatrices = new Array[BDM[Double]](layers.length)
    for(i <- (layers.size - 1) until (0, -1)){
      deltas(i) = if (i == layers.length - 1) {
        layers(i).delta(outputs(i), data)
      } else {
        layers(i).delta(outputs(i), deltas(i + 1), layers(i + 1).weights)
      }
      gradientMatrices(i) = deltas(i) * outputs(i - 1).t
      /* NB! dividing by the number of instances in
       * the batch to be transparent for the optimizer */
      gradientMatrices(i) :/= outputs(i).cols.toDouble
    }
    val avgDeltas = new Array[BDV[Double]](layers.length)
    (gradientMatrices, avgDeltas)
  }

}


