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

package org.apache.spark.mllib.ann2

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *, sum => Bsum}
import breeze.numerics.{sigmoid => Bsigmoid}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.util.random.XORShiftRandom

trait Layer {
  def getInstance(weights: Vector, position: Int): LayerModel
  def getInstance(): LayerModel
}

trait LayerModel {
  val size: Int
  def eval(data: BDM[Double]): BDM[Double]
  def delta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double]
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
  val size = w.size + b.length

  override def eval(data: BDM[Double]): BDM[Double] = {
    val output = w * data
    output(::, *) :+= b
    output
  }

  override def delta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double] = w.t * nextDelta

  override def grad(delta: BDM[Double], input: BDM[Double]): Vector = {
    val g = delta * input.t
    Vectors.dense(g.data)
  }
}

object AffineLayerModel {

  def apply(w: BDM[Double], b: BDV[Double]): AffineLayerModel = {
    new AffineLayerModel(w, b)
  }

  def unroll(weights: Vector, position: Int,
             numIn: Int, numOut: Int): (BDM[Double], BDV[Double]) = {
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
                                    activationDerivative: BDM[Double] => BDM[Double]
                                     ) extends LayerModel {
  val size = 0

  override def eval(data: BDM[Double]): BDM[Double] = activationFunction(data)

  override def delta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double] =
    nextDelta :* activationDerivative(input)

  override def grad(delta: BDM[Double], input: BDM[Double]): Vector =
    Vectors.dense(new Array[Double](0))
}

object FunctionalLayerModel {
  def apply(activationFunction: BDM[Double] => BDM[Double],
            activationDerivative: BDM[Double] => BDM[Double]): FunctionalLayerModel = {
    new FunctionalLayerModel(activationFunction, activationDerivative)
  }
}

class Topology(val layers: Array[Layer]) {

}

class FeedForwardModel(val layerModels: Array[LayerModel]) {
  def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](layerModels.length)
    outputs(0) = layerModels(0).eval(data)
    for(i <- 1 until layerModels.length){
      outputs(i) = layerModels(i).eval(outputs(i-1))
    }
    outputs
  }

  def computeGradient(data: BDM[Double], target: BDM[Double], cumGradient: Vector): Double = {
    val outputs = forward(data)
    val deltas = new Array[BDM[Double]](layerModels.length)
    val error = target - outputs.last
    val L = layerModels.length - 1
    // if last two layers form an affine + function layer == sigmoid or softmax
    if(layerModels(L).size == 0 && layerModels(L - 1).size > 0) {
      deltas(L) = error
      deltas(L - 1) = layerModels(L).delta(error, outputs(L - 1))
    } else {
      assert(false)
    }
    for(i <- (L - 2) to (0, -1)) {
      deltas(i) = layerModels(i).delta(deltas(i + 1), outputs(i))
    }
    val grads = new Array[Vector](layerModels.length)
    for(i <- 0 until layerModels.length) {
      val input = if (i==0) data else outputs(i)
      grads(i) = layerModels(i).grad(deltas(i), input)
    }
    // TODO: update cumGradient with all grads
    // TODO: take batchSize into account
    val outerError = Bsum(error :* error) / 2
    /* NB! dividing by the number of instances in
     * the batch to be transparent for the optimizer */
    outerError
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
    new FeedForwardModel(layerModels)
  }

}