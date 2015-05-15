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

import scala.collection.BitSet

import breeze.linalg.{DenseMatrix => BDM}

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.util.random.XORShiftRandom

class DropoutTopology(val layers: Array[Layer], val inputDropoutProb: Double,
                      val layerDropoutProb: Double) extends Topology {
  override def getInstance(weights: Vector): TopologyModel =
    DropoutModel(this, weights)

  override def getInstance(seed: Long): TopologyModel = DropoutModel(this, seed)
}

/* Model of Feed Forward Neural Network with drop-out.
* Implements forward, gradient computation and can return weights in vector format.
* */
class DropoutModel(val layerModels: Array[LayerModel],
                   val topology: DropoutTopology) extends TopologyModel {
  private val rand = new XORShiftRandom(System.nanoTime())

  override def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](layerModels.length)
    val lastIndex = layerModels.lastIndexWhere(lm => lm.size > 0)
    for(i <- 0 until layerModels.length){
      outputs(i) = layerModels(i).eval(if (i==0) data else outputs(i - 1))
      // use probabilities only on layers with weights except the last one
      if (topology.layerDropoutProb > 0 && layerModels(i).size > 0 && i < lastIndex) {
        outputs(i) :*= 1.0 - topology.layerDropoutProb
      }
    }
    outputs
  }

  override def computeGradient(data: BDM[Double], target: BDM[Double], cumGradient: Vector,
                      realBatchSize: Int): Double = {
    // preparing masks
    var inputMask: BitSet = null
    val layerMasks = new Array[BitSet](layerModels.length)
    if (topology.inputDropoutProb > 0) {
      inputMask = makeMask(data, topology.inputDropoutProb)
      applyMask(data, inputMask)
    }
    // forward with masks
    val lastIndex = layerModels.lastIndexWhere(lm => lm.size > 0)
    val outputs = new Array[BDM[Double]](layerModels.length)
    for (i <- 0 until layerModels.length){
      outputs(i) = layerModels(i).eval(if (i==0) data else outputs(i - 1))
      if (i < lastIndex && topology.layerDropoutProb > 0) {
        layerMasks(i) = if (layerModels(i).size > 0) {
          makeMask(outputs(i), topology.layerDropoutProb)
        } else {
          if (i==0) inputMask else layerMasks(i - 1)
        }
        applyMask(outputs(i), layerMasks(i))
      }
    }
    // error depending on output layer
    val (newE, newError) = layerModels.last match {
      case flm: FunctionalLayerModel => flm.error(outputs.last, target)
      case _ =>
        throw new UnsupportedOperationException("Non-functional layer not supported at the top")
    }
    // compute delta with masks
    val L = layerModels.length - 1
    val deltas = new Array[BDM[Double]](layerModels.length)
    deltas(L) = new BDM[Double](0, 0)
    deltas(L - 1) = newE
    for (i <- (L - 2) to (0, -1)) {
      deltas(i) = layerModels(i + 1).prevDelta(deltas(i + 1), outputs(i + 1))
      applyMask(deltas(i), layerMasks(i))
    }
    // compute gradient
    val grads = new Array[Array[Double]](layerModels.length)
    for (i <- 0 until layerModels.length) {
      val input = if (i==0) data else outputs(i - 1)
      grads(i) = layerModels(i).grad(deltas(i), input)
    }
    // update cumGradient
    val cumGradientArray = cumGradient.toArray
    var offset = 0
    // TODO: extract roll
    for (i <- 0 until grads.length) {
      val gradArray = grads(i)
      var k = 0
      while (k < gradArray.length) {
        cumGradientArray(offset + k) += gradArray(k)
        k += 1
      }
      offset += gradArray.length
    }
    newError
  }

  private def makeMask(data: BDM[Double], dropoutProb: Double): BitSet = {
    val mask = scala.collection.mutable.BitSet(data.size)
    mask(data.size) = false
    var k = 0
    while (k < data.size) {
      if (rand.nextDouble() > dropoutProb) {
        mask(k) = true
      }
      k += 1
    }
    mask
  }

  private def applyMask(data: BDM[Double], mask: BitSet): Unit = {
    var k = 0
    while (k < data.size && mask != null) {
      if (mask(k) == false) {
        data.data(k) = 0.0
      }
      k += 1
    }
  }

  override def weights(): Vector = {
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

  override def predict(data: Vector): Vector = {
    val result = forward(data.toBreeze.toDenseVector.toDenseMatrix.t)
    Vectors.dense(result.last.toArray)
  }

}

// TODO: make a fabric of models (unite with object FeedForwardModel)
object DropoutModel {
  def apply(topology: DropoutTopology, weights: Vector): DropoutModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    var offset = 0
    for(i <- 0 until layers.length){
      layerModels(i) = layers(i).getInstance(weights, offset)
      offset += layerModels(i).size
    }
    new DropoutModel(layerModels, topology)
  }

  def apply(topology: DropoutTopology, seed: Long = 11L): DropoutModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    var offset = 0
    for(i <- 0 until layers.length){
      layerModels(i) = layers(i).getInstance(seed)
      offset += layerModels(i).size
    }
    new DropoutModel(layerModels, topology)
  }
}
