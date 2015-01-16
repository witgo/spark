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

package org.apache.spark.mllib.feature

import breeze.linalg.{DenseVector => BDV, Vector => BV, DenseMatrix => BDM, Matrix => BM,
max => brzMax, Axis => brzAxis, sum => brzSum}

import org.apache.spark.mllib.neuralNetwork.NNUtil

import PartialConnectedLayer._

private[mllib] trait BaseLayer extends Serializable {
  def layerType: String

  def forward(input: BDM[Double]): BDM[Double]

  def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double])

  def outputError(output: BDM[Double], label: BDM[Double]): BDM[Double] = {
    val delta = output - label
    for (i <- 0 until delta.rows) {
      for (j <- 0 until delta.cols) {
        delta(i, j) *= computeNeuronPrimitive(output(i, j))
      }
    }
    delta
  }

  def previousError(
    input: BDM[Double],
    prevLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double]

  def computeNeuron(sum: Double): Double

  def computeNeuronPrimitive(out: Double): Double

  def computeNeuron(temp: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols) {
        temp(i, j) = computeNeuron(temp(i, j))
      }
    }
  }

  def computeNeuronPrimitive(temp: BDM[Double], output: BM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols) {
        temp(i, j) *= computeNeuronPrimitive(output(i, j))
      }
    }
  }
}


private[mllib] trait SentencePoolingLayer extends BaseLayer {
  def scaleSize: Int

  override def computeNeuron(sum: Double): Double = sum

  override def computeNeuronPrimitive(out: Double): Double = 1
}


class FixedMaxSentencePooling(scaleSize: Int)
  extends MaxSentencePooling(scaleSize) {
  override def layerType: String = "FixedMaxSentencePooling"

  override def forward(input: BDM[Double]): BDM[Double] = {
    val out = BDM.zeros[Double](scaleSize, input.cols)
    maxMatrix(input, out)
    out
  }
}

private[mllib] class MaxSentencePooling(
  val scaleSize: Int) extends SentencePoolingLayer {
  override def layerType: String = "MaxSentencePooling"

  override def forward(input: BDM[Double]): BDM[Double] = {
    val sRows = input.rows / scaleSize
    val out = BDM.zeros[Double](sRows, input.cols)
    maxMatrix(input, out)
    out
  }

  override def backward(
    input: BDM[Double],
    delta: BDM[Double]): (BDM[Double], BDV[Double]) = null

  override def previousError(
    input: BDM[Double],
    previousLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    val preDelta = input.copy
    expandMatrix(currentDelta, preDelta)
    previousLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  protected def expandMatrix(input: BM[Double], currentDelta: BM[Double]): Unit = {
    val mRows = input.rows
    val oRows = currentDelta.rows
    val cols = currentDelta.cols
    val scaleSize = mRows / oRows
    for (i <- 0 until oRows) {
      for (j <- 0 until cols) {
        val mRowsStart = i * scaleSize
        val mRowsEnd = if (i == oRows - 1) {
          math.max(mRowsStart + scaleSize, mRows)
        } else {
          mRowsStart + scaleSize
        }
        var maxVal = Double.NegativeInfinity
        var maxIndex = Int.MinValue

        for (mi <- mRowsStart until mRowsEnd) {
          val v = input(mi, j)
          if (v > maxVal) {
            maxVal = v
            maxIndex = mi
          }
        }
        for (mi <- mRowsStart until mRowsEnd) {
          if (mi == maxIndex) {
            input(mi, j) = currentDelta(i, j)
          } else {
            input(mi, j) = 0
          }
        }
      }
    }
  }

  protected def maxMatrix(matrix: BM[Double], out: BM[Double]): Unit = {
    val mRows = matrix.rows
    val oRows = out.rows
    val cols = out.cols
    val scaleSize = mRows / oRows
    for (i <- 0 until oRows) {
      for (j <- 0 until cols) {
        val mRowsStart = i * scaleSize
        val mRowsEnd = if (i == oRows - 1) {
          math.max(mRowsStart + scaleSize, mRows)
        } else {
          mRowsStart + scaleSize
        }
        var max = Double.NegativeInfinity
        for (mi <- mRowsStart until mRowsEnd) {
          val v = matrix(mi, j)
          if (v > max) {
            max = v
          }
        }
        out(i, j) = max
      }
    }
  }
}

private[mllib] class AverageSentencePooling(
  val scaleSize: Int) extends SentencePoolingLayer {
  override def layerType: String = "AverageSentencePooling"

  override def forward(input: BDM[Double]): BDM[Double] = {
    val sRows = input.rows / scaleSize
    val out = BDM.zeros[Double](sRows, input.cols)
    averageMatrix(input, out)
    out
  }

  override def backward(
    input: BDM[Double],
    delta: BDM[Double]): (BDM[Double], BDV[Double]) = null

  override def previousError(
    input: BDM[Double],
    previousLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    val preDelta = BDM.zeros[Double](input.rows, input.cols)
    expandMatrix(currentDelta, preDelta)
    previousLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  protected def expandMatrix(matrix: BM[Double], out: BM[Double]): Unit = {
    val mRows = matrix.rows
    val oRows = out.rows
    val cols = out.cols
    val scaleSize = oRows / mRows
    val remainder = oRows % mRows
    for (i <- 0 until oRows) {
      for (j <- 0 until cols) {
        val offsetRows = math.min(i / scaleSize, mRows - 1)
        val m = if (offsetRows == mRows) scaleSize + remainder else scaleSize
        out(i, j) = matrix(offsetRows, j) / m
      }
    }
  }

  protected def averageMatrix(matrix: BM[Double], out: BM[Double]): Unit = {
    val mRows = matrix.rows
    val oRows = out.rows
    val cols = out.cols
    val scaleSize = mRows / oRows
    for (i <- 0 until oRows) {
      for (j <- 0 until cols) {
        val mRowsStart = i * scaleSize
        val mRowsEnd = if (i == oRows - 1) {
          math.max(mRowsStart + scaleSize, mRows)
        } else {
          mRowsStart + scaleSize
        }
        var sum = 0.0
        for (mi <- mRowsStart until mRowsEnd) {
          sum += matrix(mi, j)
        }
        out(i, j) = sum / (mRowsEnd - mRowsStart)
      }
    }
  }
}

class FixedAverageSentencePooling(scaleSize: Int)
  extends AverageSentencePooling(scaleSize) {
  override def layerType: String = "DynamicAverageSentencePooling"

  override def forward(input: BDM[Double]): BDM[Double] = {
    val out = BDM.zeros[Double](scaleSize, input.cols)
    averageMatrix(input, out)
    out
  }
}

private[mllib] trait PartialConnectedLayer extends BaseLayer {

  def weight: BDM[Double]

  def bias: BDV[Double]

  def connTable: BDM[Double]
}

private[mllib] trait SentenceLayer extends PartialConnectedLayer {
  def inChannels: Int

  def outChannels: Int

  def connTable: BDM[Double] = BDM.ones[Double](inChannels, outChannels)

  def kernelSize: Int

  def forward(input: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val out = BDM.zeros[Double](outDim, outChannels)

    for (i <- 0 until inChannels) {
      val in = input(::, i)
      for (offset <- 0 until outDim) {
        val a = in(getInSlice(offset))
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val s: Double = getNeuron(weight, i, j).dot(a)
            out(offset, j) += s
          }
        }
      }
    }

    for (i <- 0 until outDim) {
      for (j <- 0 until outChannels) {
        out(i, j) += bias(j)
      }
    }
    computeNeuron(out)
    out
  }

  def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val gradWeight = BDM.zeros[Double](weight.rows, weight.cols)
    val gradBias = brzSum(delta, brzAxis._0).toDenseVector
    require(gradBias.length == outChannels)

    for (i <- 0 until inChannels) {
      val in = input(::, i)
      for (offset <- 0 until outDim) {
        val inA = in(getInSlice(offset))
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val z = delta(offset, j)
            val c = inA * z
            val k = getNeuron(gradWeight, i, j)
            k :+= c
          }
        }
      }
    }
    (gradWeight, gradBias)
  }


  def previousError(
    input: BDM[Double],
    prevLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val preDelta = BDM.zeros[Double](input.rows, inChannels)
    for (i <- 0 until inChannels) {
      val pv = preDelta(::, i)
      for (offset <- 0 until outDim) {
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val o = currentDelta(offset, j)
            val k = getNeuron(weight, i, j)
            val d = k * o
            pv(getInSlice(offset)) :+= d
          }
        }
      }
    }

    prevLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  @inline private def getInSlice(offset: Int): Range = {
    offset until offset + kernelSize
  }

  @inline private def getInDim(input: BDM[Double]): Int = {
    input.rows
  }

  @inline private def getOutDim(input: BDM[Double]): Int = {
    getInDim(input) - kernelSize + 1
  }

  @inline private def getNeuron(
    weight: BDM[Double],
    inOffset: Int): BDM[Double] = {
    val colOffset = outChannels * inOffset
    weight(::, colOffset until (colOffset + outChannels))
  }

  @inline private def getNeuron(
    weight: BDM[Double],
    inOffset: Int,
    outOffset: Int): BDV[Double] = {
    val colOffset = outChannels * inOffset + outOffset
    weight(::, colOffset)
  }
}


private[mllib] class ReLuSentenceLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val inChannels: Int,
  val outChannels: Int,
  val kernelSize: Int) extends SentenceLayer {

  def this(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int) {
    this(initUniformWeight(inChannels, outChannels, kernelSize,
      1, 0.0, 0.01),
      initializeBias(outChannels),
      inChannels, outChannels, kernelSize)
  }

  override def layerType: String = "ReLuSentenceInput"

  override def computeNeuron(sum: Double): Double = {
    math.max(0.0, sum)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    if (out > 0) 1.0 else 0.0
  }
}


private[mllib] class SigmoidSentenceLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val inChannels: Int,
  val outChannels: Int,
  val kernelSize: Int) extends SentenceLayer {

  def this(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int) {
    this(initUniformWeight(inChannels, outChannels, kernelSize,
      1, -4D * math.sqrt(6D / (kernelSize + inChannels * outChannels)),
      4D * math.sqrt(6D / (kernelSize + inChannels * outChannels))),
      initializeBias(outChannels),
      inChannels, outChannels, kernelSize)
  }

  override def layerType: String = "SigmoidSentence"

  override def computeNeuron(sum: Double): Double = {
    NNUtil.sigmoid(sum, 32)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    NNUtil.sigmoidPrimitive(out)
  }
}


private[mllib] class TanhSentenceLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val inChannels: Int,
  val outChannels: Int,
  val kernelSize: Int) extends SentenceLayer {

  def this(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int) {
    this(initUniformWeight(inChannels, outChannels, kernelSize,
      1, -math.sqrt(6D / (kernelSize + inChannels * outChannels)),
      math.sqrt(6D / (kernelSize + inChannels * outChannels))),
      initializeBias(outChannels),
      inChannels, outChannels, kernelSize)
  }

  override def layerType: String = "TanhSentenceLayer"

  override def computeNeuron(sum: Double): Double = {
    NNUtil.tanh(sum)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    NNUtil.tanhPrimitive(out)
  }
}

private[mllib] trait SentenceInputLayer extends SentenceLayer {

  def inChannels: Int = 1

  def windowSize: Int

  def vectorSize: Int

  override def kernelSize: Int = vectorSize * windowSize

  override def forward(input: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    require(input.rows % vectorSize == 0)
    val outDim = getOutDim(input)
    assert(outDim > 1)
    val out = BDM.zeros[Double](outDim, outChannels)
    for (offset <- 0 until outDim) {
      val a = input(getInSlice(offset), ::)
      val kernel = getNeuron(weight, 0)
      var o: BDM[Double] = a.t * kernel
      assert(o.rows == 1)
      out(offset until offset + 1, ::) :+= o
    }
    for (j <- 0 until outChannels) {
      out(::, j) += bias(j)
    }
    computeNeuron(out)
    out
  }

  override def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = {
    val outDim = getOutDim(input)
    val gradWeight = BDM.zeros[Double](weight.rows, weight.cols)
    val gradBias = brzSum(delta, brzAxis._0).toDenseVector
    require(gradBias.length == outChannels)
    for (offset <- 0 until outDim) {
      val a = input(getInSlice(offset), ::)
      val z = delta(offset until offset + 1, ::)
      val c = a * z
      val k = getNeuron(gradWeight, 0)
      k :+= c
    }
    (gradWeight, gradBias)
  }


  override def previousError(
    input: BDM[Double],
    prevLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    val outDim = getOutDim(input)
    val preDelta = BDM.zeros[Double](input.rows, inChannels)
    for (i <- 0 until inChannels) {
      val pv = preDelta(::, i)
      for (offset <- 0 until outDim) {
        for (j <- 0 until outChannels) {
          val o = currentDelta(offset, j)
          val k = getNeuron(weight, i, j)
          val d = k * o
          pv(getInSlice(offset)) :+= d
        }
      }
    }

    prevLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  @inline private def getInSlice(offset: Int): Range = {
    offset * vectorSize until (offset + windowSize) * vectorSize
  }

  @inline private def getInDim(input: BDM[Double]): Int = {
    input.rows / vectorSize
  }

  @inline private def getOutDim(input: BDM[Double]): Int = {
    getInDim(input) - windowSize + 1
  }

  @inline private def getNeuron(
    weight: BDM[Double],
    inOffset: Int): BDM[Double] = {
    val colOffset = outChannels * inOffset
    weight(::, colOffset until (colOffset + outChannels))
  }

  @inline private def getNeuron(
    weight: BDM[Double],
    inOffset: Int,
    outOffset: Int): BDV[Double] = {
    val colOffset = outChannels * inOffset + outOffset
    weight(::, colOffset)
  }
}

private[mllib] class ReLuSentenceInputLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val outChannels: Int,
  val windowSize: Int,
  val vectorSize: Int) extends SentenceInputLayer {

  def this(
    outChannels: Int,
    windowSize: Int,
    vectorSize: Int) {
    this(initUniformWeight(1, outChannels, windowSize, vectorSize,
      0.0, 0.01),
      initializeBias(outChannels),
      outChannels, windowSize, vectorSize)
  }

  override def layerType: String = "ReLuSentenceInput"

  override def computeNeuron(sum: Double): Double = {
    math.max(0.0, sum)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    if (out > 0) 1.0 else 0.0
  }
}


private[mllib] class TanhSentenceInputLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val outChannels: Int,
  val windowSize: Int,
  val vectorSize: Int) extends SentenceInputLayer {

  def this(
    outChannels: Int,
    windowSize: Int,
    vectorSize: Int) {
    this(initUniformWeight(1, outChannels, windowSize, vectorSize,
      -math.sqrt(6D / (windowSize * vectorSize + outChannels)),
      math.sqrt(6D / (windowSize * vectorSize + outChannels))),
      initializeBias(outChannels),
      outChannels, windowSize, vectorSize)
  }

  override def layerType: String = "TanhSentenceInput"

  override def computeNeuron(sum: Double): Double = {
    NNUtil.tanh(sum)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    NNUtil.tanhPrimitive(out)
  }
}

private[mllib] class SigmoidSentenceInputLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val outChannels: Int,
  val windowSize: Int,
  val vectorSize: Int) extends SentenceInputLayer {

  def this(
    outChannels: Int,
    windowSize: Int,
    vectorSize: Int) {
    this(initUniformWeight(1, outChannels, windowSize, vectorSize,
      -4D * math.sqrt(6D / (windowSize * vectorSize + outChannels)),
      4D * math.sqrt(6D / (windowSize * vectorSize + outChannels))),
      initializeBias(outChannels), outChannels, windowSize, vectorSize)
  }

  override def layerType: String = "SigmoidSentenceInput"

  override def computeNeuron(sum: Double): Double = {
    NNUtil.sigmoid(sum, 32)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    NNUtil.sigmoidPrimitive(out)
  }
}

private[mllib] object PartialConnectedLayer {
  def initializeBias(outChannels: Int): BDV[Double] = {
    BDV.zeros[Double](outChannels)
  }

  def initUniformWeight(
    inChannels: Int,
    outChannels: Int,
    windowSize: Int,
    vectorSize: Int,
    low: Double, high: Double): BDM[Double] = {
    val weight = BDM.rand[Double](windowSize * vectorSize, inChannels * outChannels)
    weight :*= (low + high)
    weight :-= low
    weight
  }
}
