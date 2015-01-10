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

import java.util.Random

import breeze.linalg.{DenseVector => BDV, Vector => BV, DenseMatrix => BDM, Matrix => BM,
max => brzMax, Axis => BrzAxis, sum => brzSum}

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{DenseMatrix => SDM, SparseMatrix => SSM, Matrix => SM,
SparseVector => SSV, DenseVector => SDV, Vector => SV, Vectors, Matrices, BLAS}
import org.apache.spark.util.Utils

import PartialConnectedLayer._

private[mllib] trait BaseLayer extends Serializable {
  def layerType: String

  def inChannels: Int

  def outChannels: Int

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
}


private[mllib] trait SentencePoolingLayer extends BaseLayer {
  def scaleSize: Int

  override def computeNeuron(sum: Double): Double = sum

  override def computeNeuronPrimitive(out: Double): Double = 1
}

private[mllib] class AverageSentencePooling(
  val inChannels: Int,
  val outChannels: Int,
  val scaleSize: Int) extends SentencePoolingLayer {
  override def layerType: String = "AveragePooling"

  override def forward(input: BDM[Double]): BDM[Double] = {
    averageMatrix(input, scaleSize, 1)
  }

  def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = null

  override def previousError(
    input: BDM[Double],
    previousLayer: BaseLayer,
    currentDelta: BDM[Double]): BDM[Double] = {
    val preDelta = expandMatrix(currentDelta, scaleSize, 1)
    for (i <- 0 until preDelta.rows) {
      for (j <- 0 until preDelta.cols) {
        preDelta(i, j) *= previousLayer.computeNeuronPrimitive(input(i, j))
      }
    }
    preDelta
  }

  private def expandMatrix(matrix: BM[Double], scaleRows: Int, scaleCols: Int): BDM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val e = BDM.zeros[Double](mRows * scaleRows, mCols * scaleCols)
    for (si <- 0 until scaleRows) {
      for (sj <- 0 until scaleCols) {
        val mRowsStart = si * mRows
        val mRowsEnd = mRowsStart + mRows
        val mColsStart = sj * mCols
        val mColsEnd = mColsStart + mCols
        e(mRowsStart until mRowsEnd, mColsStart until mColsEnd) := matrix
      }
    }
    e
  }

  private def averageMatrix(matrix: BM[Double], scaleRows: Int, scaleCols: Int): BDM[Double] = {
    scaleMatrix(matrix, scaleRows, scaleCols, m => m.valuesIterator.sum / (m.rows * m.cols))
  }

  private def scaleMatrix(
    matrix: BM[Double],
    scaleRows: Int,
    scaleCols: Int,
    fn: (BM[Double]) => Double): BDM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val sRows = mRows / scaleRows
    val sCols = mCols / scaleCols
    val s = BDM.zeros[Double](sRows, sCols)
    for (si <- 0 until sRows) {
      for (sj <- 0 until sCols) {
        val mRowsStart = si * scaleRows
        val mRowsEnd = mRowsStart + scaleRows
        val mColsStart = sj * scaleCols
        val mColsEnd = mColsStart + scaleCols
        s(si, sj) = fn(matrix(mRowsStart until mRowsEnd, mColsStart until mColsEnd))
      }
    }
    s
  }
}

private[mllib] trait PartialConnectedLayer extends BaseLayer {

  def weight: BDM[Double]

  def bias: BDV[Double]

  def connTable: BDM[Double]
}

private[mllib] trait SentenceLayer extends PartialConnectedLayer {

  def connTable: BDM[Double] = BDM.ones[Double](inChannels, outChannels)

  def kernelSize: Int = vectorSize * windowSize

  def windowSize: Int

  def vectorSize: Int

  def forward(input: BDM[Double]): BDM[Double] = {
    require(input.cols == inChannels)
    require(input.rows % vectorSize == 0)
    val inDim = this.inDim(input)
    val outDim =  this.outDim(input)
    val out = BDM.zeros[Double](outDim, outChannels)

    for (i <- 0 until inChannels) {
      val in = input(::, i)
      val o = BDM.zeros[Double](outDim, outChannels)
      for (offset <- 0 until outDim) {
        val a = in(inSlice(offset))
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val s: Double = getNeuron(weight, i, j).dot(a)
            o(offset, j) += computeNeuron(s + getBias(bias, i, j))
          }
        }
      }
      out :+= o.mapValues(computeNeuron)
    }

    out
  }

  def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = {
    require(input.cols == inChannels)
    require(input.rows % vectorSize == 0)
    val outDim = this.outDim(input)
    val gradWeight = BDM.zeros[Double](weight.rows, weight.cols)
    val gradBias = BDV.zeros[Double](inChannels * outChannels)

    for (i <- 0 until inChannels) {
      val in = input(::, i)
      val b = getBias(bias, i)
      for (offset <- 0 until outDim) {
        val inA = in(inSlice(offset))
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val z = delta(offset, j)
            val c = inA * z
            val k = getNeuron(gradWeight, i, j)
            k :+= c
            b(j) += z
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
    val outDim =  this.outDim(input)
    val preDelta = BDM.zeros[Double](input.rows, inChannels)
    for (i <- 0 until inChannels) {
      val pv = preDelta(::, i)
      for (offset <- 0 until outDim) {
        for (j <- 0 until outChannels) {
          if (connTable(i, j) > 0.0) {
            val o = currentDelta(i, j)
            val k = getNeuron(weight, i, j)
            val d = k * o
            pv(inSlice(offset)) :+= d
          }
        }
      }
    }
    for (i <- 0 until preDelta.rows) {
      for (j <- 0 until preDelta.cols) {
        preDelta(i, j) *= prevLayer.computeNeuronPrimitive(input(i, j))
      }
    }
    preDelta
  }

  @inline def inSlice(offset: Int): Range = {
    offset * vectorSize until (offset + windowSize) * vectorSize
  }

  @inline private def inDim(input: BDM[Double]): Int = {
    input.rows
  }

  @inline private def outDim(input: BDM[Double]): Int = {
    inDim(input) - windowSize + 1
  }

  @inline private def getBias(
    bias: BDV[Double],
    inOffset: Int): BDV[Double] = {
    val offset = outChannels * inOffset
    bias(offset until (offset + outChannels))
  }

  @inline private def getBias(
    bias: BDV[Double],
    inOffset: Int,
    outOffset: Int): Double = {
    val offset = outChannels * inOffset + outOffset
    bias(offset)
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

private[mllib] class ReLUSentenceLayer(
  val weight: BDM[Double],
  val bias: BDV[Double],
  val inChannels: Int,
  val outChannels: Int,
  val windowSize: Int,
  val vectorSize: Int) extends SentenceLayer {

  def this(
    inChannels: Int,
    outChannels: Int,
    windowSize: Int,
    vectorSize: Int) {
    this(initUniformWeight(inChannels, outChannels, windowSize, vectorSize),
      initializeBias(outChannels), inChannels, outChannels, windowSize, vectorSize)
  }

  override def layerType: String = "ReLUSentence"

  override def computeNeuron(sum: Double): Double = {
    math.max(0.0, sum)
  }

  override def computeNeuronPrimitive(out: Double): Double = {
    if (out > 0) 1.0 else 0.0
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
    vectorSize: Int): BDM[Double] = {
    val weight = BDM.rand[Double](windowSize * vectorSize, inChannels * outChannels)
    weight :*= 0.01
    weight
  }
}
