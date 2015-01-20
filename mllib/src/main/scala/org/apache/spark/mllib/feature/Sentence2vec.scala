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
max => brzMax, Axis => brzAxis, sum => brzSum, axpy => brzAxpy}

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.neuralNetwork.{NNUtil, MLP}
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

@Experimental
class Sentence2vec(
  val sentenceLayer: Array[BaseLayer],
  val mlp: MLP,
  val vectorSize: Int,
  val windowSize: Int) extends Serializable with Logging {

  @transient var word2Vec: BDV[Double] = null
  @transient private lazy val numSentenceLayer = sentenceLayer.length
  @transient private lazy val numLayer = numSentenceLayer + mlp.numLayer
  @transient private lazy val rand: Random = new XORShiftRandom()
  @transient private lazy val wordSize: Int = word2Vec.length / vectorSize
  @transient private lazy val inputWindow: Int = sentenceLayer.head.
    asInstanceOf[SentenceInputLayer].windowSize

  def setWord2Vec(word2Vec: BDV[Double]): Unit = {
    this.word2Vec = word2Vec
  }

  def predict(sentence: Array[Int]): BDV[Double] = {
    var input = sentenceToInput(sentence)
    for (i <- 0 until sentenceLayer.length) {
      input = sentenceLayer(i).forward(input)
    }
    input.toDenseVector
  }

  protected def computeGradient(
    sentence: Array[Int]): (Array[(BDM[Double], BDV[Double])], Double) = {
    val in = sentenceToInput(sentence)
    val sentOutput = sentenceComputeOutputs(sentence, in)
    val (prevDelta, mlpGrad, cost) = mlpComputeGradient(sentence, in,
      sentOutput.last.toDenseVector)
    val sentGrad = sentenceComputeGradient(sentence, in, prevDelta, sentOutput)
    (sentGrad ++ mlpGrad, cost)
  }

  protected[mllib] def mlpComputeGradient(
    sentence: Array[Int],
    sentVec: BDM[Double],
    sentOut: BDV[Double]) = {
    if (rand.nextDouble() < 0.001) {
      println(s"sentOut: ${sentOut.valuesIterator.map(_.abs).sum / sentOut.length}")
    }
    val sentenceSize = sentence.size
    val mlpIn = BDM.zeros[Double](vectorSize, 1)
    val label = BDM.zeros[Double](wordSize, 1)
    label := (0.1 / wordSize)
    for (pos <- 0 until sentenceSize) {
      val word = sentence(pos)
      mlpIn(::, 0) := sentOut
      label(word, 0) += (0.9 / sentenceSize)
    }
    val (mlpOuts, mlpDeltas) = mlp.computeDelta(mlpIn, label)
    val mlpGrads = mlp.computeGradientGivenDelta(mlpIn, mlpOuts, mlpDeltas)
    val cost = NNUtil.crossEntropy(mlpOuts.last, label)
    val prevDelta: BDM[Double] = mlp.innerLayers.head.weight.t * mlpDeltas.head
    val inputDelta = prevDelta(::, 0)
    if (rand.nextDouble() < 0.001) {
      println(s"mlpDelta: ${inputDelta.valuesIterator.map(_.abs).sum / vectorSize}")
    }
    assert(inputDelta.length == vectorSize)

    (inputDelta, mlpGrads, cost)
  }

  protected[mllib] def mlpComputeGradient2(
    sentence: Array[Int],
    sentVec: BDM[Double],
    sentOut: BDV[Double]) = {
    if (rand.nextDouble() < 0.001) {
      println(s"sentOut: ${sentOut.valuesIterator.map(_.abs).sum / sentOut.length}")
    }
    val sentenceSize = sentence.size
    val randomize = Utils.randomize(0 until sentenceSize).slice(0, 1)
    val randomizeSize = randomize.size
    val mlpIn = BDM.zeros[Double](vectorSize, randomizeSize)
    val label = BDM.zeros[Double](wordSize, randomizeSize)
    label := 0.1 / wordSize
    for (i <- 0 until randomizeSize) {
      val pos = randomize(i)
      val word = sentence(pos)
      val b = rand.nextInt(windowSize)
      var s = 0.0
      val sum = mlpIn(::, i)
      var a = b
      while (a < windowSize * 2 + 1 - b) {
        if (a != windowSize) {
          val c = pos - windowSize + a
          if (c >= 0 && c < sentenceSize) {
            val lastWord = sentence(c)
            sum :+= wordToVector(lastWord)
            s += 1
          }
        }
        a += 1
      }
      assert(s > 0)
      sum :/= s
      sum :+= sentOut
      label(word, i) += 0.9
    }
    val (mlpOuts, mlpDeltas) = mlp.computeDelta(mlpIn, label)
    val mlpGrads = mlp.computeGradientGivenDelta(mlpIn, mlpOuts, mlpDeltas)
    val cost = NNUtil.crossEntropy(mlpOuts.last, label) / randomizeSize
    val prevDelta: BDM[Double] = mlp.innerLayers.head.weight.t * mlpDeltas.head
    val inputDelta = brzSum(prevDelta, brzAxis._1)
    inputDelta :/= randomizeSize.toDouble
    if (rand.nextDouble() < 0.001) {
      println(s"inputDelta: ${inputDelta.valuesIterator.map(_.abs).sum / vectorSize}")
    }
    assert(inputDelta.length == vectorSize)

    (inputDelta, mlpGrads, cost)
  }

  protected[mllib] def sentenceComputeOutputs(
    sentence: Array[Int],
    x: BDM[Double]): Array[BDM[Double]] = {
    val output = new Array[BDM[Double]](numSentenceLayer)
    for (i <- 0 until numSentenceLayer) {
      output(i) = sentenceLayer(i).forward(if (i == 0) x else output(i - 1))
    }
    output
  }

  protected[mllib] def sentenceComputeGradient(
    sentence: Array[Int],
    x: BDM[Double],
    mlpDelta: BDV[Double],
    output: Array[BDM[Double]]): Array[(BDM[Double], BDV[Double])] = {
    val delta = new Array[BDM[Double]](numSentenceLayer)
    val prevDelta = new BDM[Double](output.last.rows, output.last.cols, mlpDelta.toArray)
    for (i <- (0 until numSentenceLayer).reverse) {
      val out = output(i)
      val currentLayer = sentenceLayer(i)
      delta(i) = if (i == numSentenceLayer - 1) {
        prevDelta
      } else {
        val nextLayer = sentenceLayer(i + 1)
        val nextDelta = delta(i + 1)
        nextLayer.previousError(out, currentLayer, nextDelta)
      }
    }

    val grads = new Array[(BDM[Double], BDV[Double])](numSentenceLayer)
    for (i <- 0 until numSentenceLayer) {
      grads(i) = sentenceLayer(i).backward(if (i == 0) x else output(i - 1), delta(i))
    }
    grads
  }

  private def wordToVector(pos: Int): BDV[Double] = {
    word2Vec(pos * vectorSize until (pos + 1) * vectorSize)
  }

  private[mllib] def sentenceToInput(sentence: Array[Int]): BDM[Double] = {
    val vectors = sentence.map { s =>
      word2Vec(s * vectorSize until (s + 1) * vectorSize)
    }
    BDV.vertcat(vectors.toArray: _*).asDenseMatrix.t
  }
}

object Sentence2vec {
  def train[S <: Iterable[String]](
    dataset: RDD[S],
    word2VecModel: Word2VecModel,
    numIter: Int,
    learningRate: Double,
    fraction: Double): (Sentence2vec, BDV[Double], Map[String, Int]) = {
    val wordVectors = word2VecModel.getVectors
    val vectorSize = wordVectors.head._2.size
    val word2Index = wordVectors.keys.zipWithIndex.toMap
    val sentences = dataset.map(_.filter(w => word2Index.contains(w))).filter(_.size > 4).
      map(w => w.map(t => word2Index(t)).toArray)
    val word2Vec = BDV.zeros[Double](vectorSize * wordVectors.size)
    wordVectors.foreach { case (word, f) =>
      val offset = word2Index(word) * vectorSize
      for (i <- 0 until f.length) {
        word2Vec(offset + i) = f(i).toDouble
      }
    }

    val sentenceLayer: Array[BaseLayer] = new Array[BaseLayer](4)
    sentenceLayer(0) = new ReLuSentenceInputLayer(16, 7, vectorSize)
    sentenceLayer(1) = new DynamicKMaxSentencePooling(6, 0.5)
    val layer = new ReLuSentenceLayer(16, vectorSize / 4, 5)
    if (layer.outChannels > 1 && layer.inChannels > 1) {
      val s = layer.outChannels / 2
      for (i <- 0 until layer.inChannels) {
        for (j <- 0 until s) {
          val offset = (i + j) % layer.outChannels
          layer.connTable(i, offset) = 0.0
        }
      }
    }
    sentenceLayer(2) = layer
    sentenceLayer(3) = new KMaxSentencePooling(4)
    val mlp = new MLP(Array(vectorSize, 32, word2Index.size), 0.5, 0.5)
    val sent2vec = new Sentence2vec(sentenceLayer, mlp, vectorSize, 5)
    val wordBroadcast = dataset.context.broadcast(word2Vec)
    val momentumSum = new Array[(BDM[Double], BDV[Double])](sent2vec.numLayer)
    for (iter <- 0 until numIter) {
      val sentBroadcast = dataset.context.broadcast(sent2vec)
      val (gradientSum, lossSum, miniBatchSize) = trainOnce(sentences,
        sentBroadcast, wordBroadcast, iter, fraction)
      if (Utils.random.nextDouble() < 0.05) {
        sentenceLayer.zipWithIndex.foreach { case (b, i) =>
          b match {
            case s: SentenceLayer =>
              val weight = s.weight
              println(s"sentenceLayer weight $i: " +
                weight.valuesIterator.map(_.abs).sum / weight.size)
            case _ =>
          }
        }
      }

      if (miniBatchSize > 0) {
        gradientSum.filter(t => t != null).foreach(m => {
          m._1 :/= miniBatchSize.toDouble
          m._2 :/= miniBatchSize.toDouble
        })
        println(s"loss $iter : " + (lossSum / miniBatchSize))
        updateParameters(momentumSum, gradientSum, sent2vec, iter, learningRate)
      }
      sentBroadcast.destroy()
    }
    (sent2vec, word2Vec, word2Index)
  }

  def updateParameters(
    momentumSum: Array[(BDM[Double], BDV[Double])],
    gradSum: Array[(BDM[Double], BDV[Double])],
    sent2Vec: Sentence2vec,
    iter: Int,
    learningRate: Double): Unit = {
    val momentum = if (iter < 32) {
      math.min(math.pow(1.1, iter) * 0.5, 0.9)
    } else {
      0.9
    }
    val numSentenceLayer = sent2Vec.numSentenceLayer
    mergerParameters(momentumSum, gradSum, momentum)
    for (i <- 0 until numSentenceLayer) {
      if (momentumSum(i) != null) {
        val layer = sent2Vec.sentenceLayer(i).asInstanceOf[PartialConnectedLayer]
        brzAxpy(-learningRate, momentumSum(i)._1, layer.weight)
        brzAxpy(-learningRate, momentumSum(i)._2, layer.bias)
      }
    }

    for (i <- 0 until sent2Vec.mlp.numLayer) {
      val layer = sent2Vec.mlp.innerLayers(i)
      brzAxpy(-learningRate, momentumSum(numSentenceLayer + i)._1, layer.weight)
      brzAxpy(-learningRate, momentumSum(numSentenceLayer + i)._2, layer.bias)
    }
  }

  def mergerParameters(
    a: Array[(BDM[Double], BDV[Double])],
    b: Array[(BDM[Double], BDV[Double])],
    momentum: Double = 1.0): Unit = {
    for (i <- 0 until a.length) {
      if (a(i) == null) {
        a(i) = b(i)
      } else if (b(i) != null) {
        if (momentum < 1.0) {
          a(i)._1 :*= momentum
          a(i)._2 :*= momentum
        }
        a(i)._1 :+= b(i)._1
        a(i)._2 :+= b(i)._2
      }
    }
  }

  def trainOnce(
    dataset: RDD[Array[Int]],
    sent2Vec: Broadcast[Sentence2vec],
    word2Vec: Broadcast[BDV[Double]],
    iter: Int,
    fraction: Double): (Array[(BDM[Double], BDV[Double])], Double, Long) = {
    dataset.context.broadcast()
    val numLayer = sent2Vec.value.numLayer
    dataset.sample(false, fraction).treeAggregate((new Array[(BDM[Double],
      BDV[Double])](numLayer), 0D, 0L))(seqOp = (c, v) => {
      sent2Vec.value.setWord2Vec(word2Vec.value)
      val g = sent2Vec.value.computeGradient(v)
      mergerParameters(c._1, g._1)
      (c._1, c._2 + g._2, c._3 + 1)
    }, combOp = (c1, c2) => {
      mergerParameters(c1._1, c2._1)
      (c1._1, c1._2 + c2._2, c1._3 + c2._3)
    })
  }
}
