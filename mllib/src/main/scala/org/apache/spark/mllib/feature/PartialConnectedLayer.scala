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


private[mllib] class MaxSentencePooling(
  minK_ : Int)
  extends DynamicMaxSentencePooling(minK_, 1.0) {
  override def layerType: String = "FixedMaxSentencePooling"

  override def forward(input: BDM[Double]): BDM[Double] = {
    assert(input.rows >= minK)
    val out = BDM.zeros[Double](minK, input.cols)
    maxMatrix(input, out)
    out
  }
}

private[mllib] class DynamicMaxSentencePooling(
  val minK: Int,
  val scale: Double) extends BaseLayer {
  override def layerType: String = "MaxSentencePooling"

  override def computeNeuron(sum: Double): Double = sum

  override def computeNeuronPrimitive(out: Double): Double = 1

  override def forward(input: BDM[Double]): BDM[Double] = {
    assert(input.rows >= minK)
    val sRows = math.max(minK, (input.rows * scale).floor.toInt)
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
    expandMatrix(preDelta, currentDelta)
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
        assert(maxIndex > -1)
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

private[mllib] class DynamicKMaxSentencePooling(
  val minK: Int,
  val scale: Double) extends BaseLayer {
  override def layerType: String = "DynamicKMaxSentencePooling"

  override def computeNeuron(sum: Double): Double = sum

  override def computeNeuronPrimitive(out: Double): Double = 1

  override def forward(input: BDM[Double]): BDM[Double] = {
    val K = math.max(minK, (input.rows * scale).ceil.toInt)
    assert(input.rows >= K)
    val out = BDM.zeros[Double](K, input.cols)
    maxMatrix(input, out)
    out
  }

  protected def maxMatrix(input: BDM[Double], out: BDM[Double]): Unit = {
    val K = out.rows
    for (i <- 0 until input.cols) {
      val row = input(::, i)
      val pair = row.iterator.toArray.sortBy(t => -t._2).slice(0, K).sortBy(t => t._1)
      for (j <- 0 until K) {
        out(j, i) = pair(j)._2
      }
    }
  }

  override def backward(
    input: BDM[Double],
    delta: BDM[Double]): (BDM[Double], BDV[Double]) = null

  override def previousError(
    input: BDM[Double],
    previousLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    val preDelta = input.copy
    expandMatrix(preDelta, currentDelta)
    previousLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  protected def expandMatrix(input: BDM[Double], currentDelta: BDM[Double]): Unit = {
    val K = currentDelta.rows
    assert(input.rows >= K)
    for (i <- 0 until input.cols) {
      val row = input(::, i)
      val pair = row.iterator.toArray.sortBy(t => -t._2).slice(0, K).sortBy(t => t._1)
      row := 0.0
      for (j <- 0 until K) {
        row(pair(j)._1) = currentDelta(j, i)
      }
    }
  }
}

private[mllib] class KMaxSentencePooling(K: Int)
  extends BaseLayer {
  override def layerType: String = "KMaxSentencePooling"

  override def computeNeuron(sum: Double): Double = sum

  override def computeNeuronPrimitive(out: Double): Double = 1

  override def forward(input: BDM[Double]): BDM[Double] = {
    val out = BDM.zeros[Double](K, input.cols)
    maxMatrix(input, out)
    out
  }

  protected def maxMatrix(input: BDM[Double], out: BDM[Double]): Unit = {
    assert(out.rows == K)
    assert(input.rows >= K)
    for (i <- 0 until input.cols) {
      val row = input(::, i)
      val pair = row.iterator.toArray.sortBy(t => -t._2).slice(0, K).sortBy(t => t._1)
      for (j <- 0 until K) {
        out(j, i) = pair(j)._2
      }
    }
  }

  override def backward(
    input: BDM[Double],
    delta: BDM[Double]): (BDM[Double], BDV[Double]) = null

  override def previousError(
    input: BDM[Double],
    previousLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    val preDelta = input.copy
    expandMatrix(preDelta, currentDelta)
    previousLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  protected def expandMatrix(input: BDM[Double], currentDelta: BDM[Double]): Unit = {
    assert(currentDelta.rows == K)
    assert(input.rows >= K)
    for (i <- 0 until input.cols) {
      val row = input(::, i)
      val pair = row.iterator.toArray.sortBy(t => -t._2).slice(0, K).sortBy(t => t._1)
      row := 0.0
      for (j <- 0 until K) {
        row(pair(j)._1) = currentDelta(j, i)
      }
    }
  }
}

private[mllib] trait PartialConnectedLayer extends BaseLayer {

  def weight: BDM[Double]

  def bias: BDV[Double]

  val connTable: BDM[Double]
}


private[mllib] trait NarrowSentenceLayer extends PartialConnectedLayer {
  def inChannels: Int

  def outChannels: Int

  val connTable: BDM[Double] = BDM.ones[Double](inChannels, outChannels)

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
            val kernel = getNeuron(weight, i, j)
            val s: Double = kernel.dot(a)
            out(offset, j) += s
          }
        }
      }
    }

    for (j <- 0 until outChannels) {
      out(::, j) :+= bias(j)
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
        val a = in(getInSlice(offset))
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val z = delta(offset, j)
            val kernel = getNeuron(gradWeight, i, j)
            kernel :+= (a * z)
          }
        }
      }
    }
    (gradWeight, gradBias)
  }

  def previousError(
    input: BDM[Double],
    prevLayer: BaseLayer,
    delta: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val preDelta = BDM.zeros[Double](input.rows, inChannels)
    for (i <- 0 until inChannels) {
      val pa = preDelta(::, i)
      for (offset <- 0 until outDim) {
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val o = delta(offset, j)
            val kernel = getNeuron(weight, i, j)
            pa(getInSlice(offset)) :+= (kernel * o)
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

  @inline private def getOutDim(input: BDM[Double]): Int = {
    // Narrow convolution
    input.rows - kernelSize + 1
  }

  @inline private def getNeuron(
    weight: BDM[Double],
    inOffset: Int,
    outOffset: Int): BDV[Double] = {
    val colOffset = outChannels * inOffset + outOffset
    weight(::, colOffset)
  }
}

private[mllib] trait SentenceLayer extends PartialConnectedLayer {
  def inChannels: Int

  def outChannels: Int

  lazy val connTable: BDM[Double] = BDM.ones[Double](inChannels, outChannels)

  def kernelSize: Int

  def forward(input: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val out = BDM.zeros[Double](outDim, outChannels)
    val in = BDV.zeros[Double](kernelSize)
    for (offset <- 0 until outDim) {
      for (i <- 0 until inChannels) {
        setIn(offset, i, input, in)
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val kernel = getNeuron(weight, i, j)
            val d = in.dot(kernel)
            out(offset, j) += d
          }
        }
      }
    }
    for (j <- 0 until outChannels) {
      out(::, j) += bias(j)
    }
    computeNeuron(out)
    out
  }

  def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val gradWeight = BDM.zeros[Double](weight.rows, weight.cols)
    val gradBias = brzSum(delta, brzAxis._0).toDenseVector
    val in = BDV.zeros[Double](kernelSize)
    require(gradBias.length == outChannels)
    for (offset <- 0 until outDim) {
      for (i <- 0 until inChannels) {
        setIn(offset, i, input, in)
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val z = delta(offset, j)
            val kernel = getNeuron(gradWeight, i, j)
            kernel :+= (in * z)
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
    for (offset <- 0 until outDim) {
      for (i <- 0 until inChannels) {
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val d = currentDelta(offset, j)
            val kernel = getNeuron(weight, i, j)
            val err: BDV[Double] = kernel * d
            for (i <- 0 until kernelSize) {
              val s = offset + i - (kernelSize - 1)
              if (s >= 0 && s < input.rows) {
                preDelta(s, i) += err(i)
              }
            }
          }
        }
      }
    }

    prevLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

  @inline private def getOutDim(input: BDM[Double]): Int = {
    // Wide convolution
    input.rows + kernelSize - 1
  }

  @inline private def setIn(
    offset: Int,
    channel: Int,
    input: BDM[Double],
    in: BDV[Double]): Unit = {
    for (i <- 0 until kernelSize) {
      val s = offset + i - (kernelSize - 1)
      in(i) = if (s >= 0 && s < input.rows) {
        input(s, channel)
      }
      else {
        0.0
      }
    }
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
      1, -math.sqrt(6D / (kernelSize * inChannels + outChannels)),
      math.sqrt(6D / (kernelSize * inChannels + outChannels))),
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


private[mllib] class SoftPlusSentenceLayer(
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

  override def layerType: String = "SoftPlusSentenceInput"

  override def computeNeuron(sum: Double): Double = {
    NNUtil.softplus(sum, 32)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    NNUtil.softplusPrimitive(out, 32)
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
    val in = BDV.zeros[Double](windowSize * vectorSize)
    for (offset <- 0 until outDim) {
      for (i <- 0 until input.cols) {
        setIn(offset, i, input, in)
        for (j <- 0 until outChannels) {
          val kernel = getNeuron(weight, i, j)
          val d = in.dot(kernel)
          out(offset, j) += d
        }
      }
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
    val in = BDV.zeros[Double](windowSize * vectorSize)
    require(gradBias.length == outChannels)
    for (offset <- 0 until outDim) {
      for (i <- 0 until input.cols) {
        setIn(offset, i, input, in)
        for (j <- 0 until outChannels) {
          val kernel = getNeuron(gradWeight, i, j)
          val d = delta(offset, j)
          kernel :+= (in :* d)
        }
      }
    }
    (gradWeight, gradBias)
  }

  override def previousError(
    input: BDM[Double],
    prevLayer: BaseLayer,
    delta: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    val outDim = getOutDim(input)
    val preDelta = BDM.zeros[Double](input.rows, inChannels)
    for (offset <- 0 until outDim) {
      for (i <- 0 until inChannels) {
        for (j <- 0 until outChannels) {
          val d = delta(offset, j)
          val kernel = getNeuron(weight, i, j)
          val err: BDV[Double] = kernel * d
          for (i <- 0 until windowSize) {
            val s = offset + i - (windowSize - 1)
            if (s >= 0 && s < input.rows / vectorSize) {
              preDelta(s * vectorSize until (s + 1) * vectorSize, i) :+=
                err(i * vectorSize until (i + 1) * vectorSize)
            }
          }
        }
      }
    }

    prevLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
    // throw new NotImplementedError("previousError is not implemented.")
  }

  @inline private def setIn(
    offset: Int,
    channel: Int,
    input: BDM[Double],
    in: BDV[Double]): Unit = {
    for (i <- 0 until windowSize) {
      val s = offset + i - (windowSize - 1)
      if (s >= 0 && s < input.rows / vectorSize) {
        in(i * vectorSize until (i + 1) * vectorSize) :=
          input(s * vectorSize until (s + 1) * vectorSize, channel)
      }
      else {
        in(i * vectorSize until (i + 1) * vectorSize) := 0.0
      }
    }
  }

  @inline private def getOutDim(input: BDM[Double]): Int = {
    // Wide convolution
    input.rows / vectorSize + windowSize - 1
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


private[mllib] class SoftPlusSentenceInputLayer(
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

  override def layerType: String = "SoftPlusSentenceInput"

  override def computeNeuron(sum: Double): Double = {
    NNUtil.softplus(sum, 32)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    NNUtil.softplusPrimitive(out, 32)
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
    low: Double,
    high: Double): BDM[Double] = {
    val weight = BDM.rand[Double](windowSize * vectorSize, inChannels * outChannels)
    weight :*= high - low
    weight :+= low
    weight
  }
}
