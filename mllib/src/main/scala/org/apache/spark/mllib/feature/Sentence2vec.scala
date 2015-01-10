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

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, Matrix => BM,
max => brzMax, Axis => BrzAxis, sum => brzSum}

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{DenseMatrix => SDM, SparseMatrix => SSM, Matrix => SM,
SparseVector => SSV, DenseVector => SDV, Vector => SV, Vectors, Matrices, BLAS}
import org.apache.spark.mllib.neuralNetwork.{NNUtil, MLP}
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

@Experimental
class Sentence2vec(
  val sentenceLayer: Array[BaseLayer],
  val mlp: MLP,
  val word2Vec: BDV[Double],
  val vectorSize: Int,
  val windowSize: Int) extends Serializable with Logging {

  @transient private lazy val numSentenceLayer = sentenceLayer.length
  @transient private lazy val numLayer = numSentenceLayer + mlp.numLayer
  @transient private lazy val inputLayer = sentenceLayer.head.asInstanceOf[SentenceLayer]
  @transient private lazy val rand: Random = new XORShiftRandom()
  @transient private lazy val wordSize: Int = word2Vec.length / vectorSize
  require(sentenceLayer.last.outChannels == 1)

  def predict(sentence: Array[Int]): BDV[Double] = {
    var input = toInput(sentence)
    for (i <- 0 until sentenceLayer.length) {
      input = sentenceLayer(i).forward(input)
    }
    val out = mlp.predict(input)
    out(::, 0)
  }

  protected def computeGradient(
    sentence: Array[Int]): (Array[(BDM[Double],
    BDV[Double])], Double) = {
    val input = toInput(sentence)
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)
    val (out, delta, cost) = computeDelta(sentence, input)
    for (i <- 0 until numSentenceLayer) {
      grads(i) = sentenceLayer(i).backward(if (i == 0) input else out(i - 1), delta(i))
    }

    for (i <- 0 until mlp.numLayer) {
      val input = out(numSentenceLayer + i - 1)
      val m = mlp.innerLayers(i).backward(input, delta(i))
      grads(numSentenceLayer + i) = (m._1, m._2)
    }
    (grads, cost)
  }

  protected[mllib] def computeDelta(
    sentence: Array[Int],
    input: BDM[Double]): (Array[BDM[Double]], Array[BDM[Double]], Double) = {
    val output = new Array[BDM[Double]](numLayer)
    val delta = new Array[BDM[Double]](numLayer)
    for (i <- 0 until numSentenceLayer) {
      output(i) = sentenceLayer(i).forward(if (i == 0) input else output(i - 1))
    }

    val sentenceOut = output(numSentenceLayer - 1)
    val sentenceSize = sentence.size
    val x = BDM.zeros[Double](vectorSize, sentenceSize)
    val label = BDM.zeros[Double](wordSize, sentenceSize)
    for (pos <- 0 until sentenceSize) {
      val word = sentence(pos)
      val b = rand.nextInt(windowSize)
      var s = 0.0
      val sum = BDV.zeros[Double](vectorSize)
      var a = b
      while (a < windowSize * 2 + 1 - b) {
        if (a != windowSize) {
          val c = pos - windowSize + a
          if (c >= 0 && c < sentenceSize) {
            val lastWord = sentence(c)
            sum :+= toVector(lastWord)
            s += 1
          }
        }
        a += 1
      }
      sum :/= s
      sum :*= sentenceOut(::, 0)
      x(::, pos) := sum
      label(word, pos) = 1.0
    }

    val m = mlp.computeDelta(sentenceOut, label)
    for (i <- 0 until mlp.numLayer) {
      output(numSentenceLayer + i) = m._1(i)
      delta(numSentenceLayer + i) = m._2(i)
    }
    val cost = NNUtil.crossEntropy(output.last, label)

    var prevDelta: BDM[Double] = mlp.innerLayers.head.weight.t * delta(numSentenceLayer)
    for (pos <- 0 until sentenceSize) {
      val df = x(::, pos) :/ sentenceOut(::, 0)
      prevDelta(::, pos) :*= df
    }
    prevDelta = brzSum(prevDelta, BrzAxis._1).asDenseMatrix.t.toDenseMatrix

    for (i <- (0 until numSentenceLayer).reverse) {
      val out = output(i)
      val currentLayer = sentenceLayer(i)
      delta(i) = if (i == numSentenceLayer - 1) {
        prevDelta
      }
      else {
        val nextLayer = sentenceLayer(i + 1)
        val nextDelta = delta(i + 1)
        nextLayer.previousError(out, currentLayer, nextDelta)
      }
    }
    (output, delta, cost)
  }

  private def toVector(pos: Int): BDV[Double] = {
    word2Vec(pos * vectorSize until (pos + 1) * vectorSize)
  }

  private def toInput(sentence: Array[Int]): BDM[Double] = {
    val vectors = sentence
      .map(s => word2Vec(s * vectorSize until (s + 1) * vectorSize))
    BDV.vertcat(vectors.toArray: _*).asDenseMatrix.t
  }
}
