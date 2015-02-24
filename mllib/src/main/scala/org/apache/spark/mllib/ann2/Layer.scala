package org.apache.spark.mllib.ann2

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *}
import breeze.numerics.{sigmoid => Bsigmoid}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.util.random.XORShiftRandom

trait Layer {
  def getInstance(weights: Vector, position: Int): LayerModel
  def getInstance(): LayerModel
}

trait LayerModel {
  def eval(data: BDM[Double]): BDM[Double]
  def prevDelta(delta: BDM[Double], input: BDM[Double]): BDM[Double]
  def grad(delta: BDM[Double], input: BDM[Double]): Vector
}

class AffineLayer(numIn: Int, numOut: Int) extends Layer {
  override def getInstance(weights: Vector, position: Int): LayerModel = {
    val (w, b) = AffineLayerModel.unroll(weights, position, numIn, numOut)
    AffineLayerModel(w, b)
  }

  override def getInstance(): LayerModel = {
    val (w, b) = AffineLayerModel.randomWeights(numIn, numOut)
    AffineLayerModel(w, b)
  }
}

class AffineLayerModel private(w: BDM[Double], b: BDV[Double]) extends LayerModel {
  override def eval(data: BDM[Double]): BDM[Double] = {
    val output = w * data
    output(::, *) :+= b
    output
  }

  override def prevDelta(delta: BDM[Double], input: BDM[Double]): BDM[Double] = w.t * delta

  override def grad(delta: BDM[Double], input: BDM[Double]): Vector = {
    val g = delta * input.t
    Vectors.dense(g.data)
  }
}

object AffineLayerModel {

  def apply(w: BDM[Double], b: BDV[Double]): AffineLayerModel = {
    new AffineLayerModel(w, b)
  }

  def unroll(weights: Vector, position: Int, numIn: Int, numOut: Int): (BDM[Double], BDV[Double]) = {
    val weightsCopy = weights.toArray
    val w = new BDM[Double](numOut, numIn, weightsCopy, position)
    val b = new BDV[Double](weightsCopy, position + (numOut * numIn), 1, numOut)
    (w, b)
  }

  def randomWeights(numIn: Int, numOut: Int, seed: Long = 11L): (BDM[Double], BDV[Double]) = {
    val rand: XORShiftRandom = new XORShiftRandom(seed)
    val weights = BDM.fill[Double](numOut, numIn){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    val bias = BDV.fill[Double](numOut){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    (weights, bias)
  }
}

object ANNFunctions {

  def Sigmoid(data: BDM[Double]): BDM[Double] = Bsigmoid(data)

  def SigmoidDerivative(data: BDM[Double]): BDM[Double] = {
    val derivative = BDM.ones[Double](data.rows, data.cols)
    derivative :-= data
    derivative :*= data
    derivative
  }
}

class FunctionalLayer (activationFunction: BDM[Double] => BDM[Double],
                       activationDerivative: BDM[Double] => BDM[Double]) extends Layer {
  override def getInstance(weights: Vector, position: Int): LayerModel = getInstance()

  override def getInstance(): LayerModel =
    FunctionalLayerModel(activationFunction, activationDerivative)
}

class FunctionalLayerModel private (activationFunction: BDM[Double] => BDM[Double],
                                    activationDerivative: BDM[Double] => BDM[Double]) extends LayerModel {
  override def eval(data: BDM[Double]): BDM[Double] = activationFunction(data)

  override def prevDelta(delta: BDM[Double], input: BDM[Double]): BDM[Double] = delta :* activationDerivative(input)

  override def grad(delta: BDM[Double], input: BDM[Double]): Vector = Vectors.dense(new Array[Double](0))
}

object FunctionalLayerModel {
  def apply(activationFunction: BDM[Double] => BDM[Double],
            activationDerivative: BDM[Double] => BDM[Double]): FunctionalLayerModel = {
    new FunctionalLayerModel(activationFunction, activationDerivative)
  }
}