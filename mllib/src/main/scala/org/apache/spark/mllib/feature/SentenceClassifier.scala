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
import org.apache.spark.mllib.neuralNetwork._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

@Experimental
class SentenceClassifier(
  val sentenceLayer: Array[BaseLayer],
  val mlp: MLP,
  val vectorSize: Int) extends Serializable with Logging {

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
    val mlpIn = BDM.zeros[Double](input.size, 1)
    mlpIn(::, 0) := input.toDenseVector
    val mlpOut = mlp.predict(mlpIn)
    mlpOut(::, 0)
  }

  protected def computeGradient(
    sentence: Array[Int],
    label: BDV[Double]): (Array[(BDM[Double], BDV[Double])], Double) = {
    val in = sentenceToInput(sentence)
    val sentOutput = sentenceComputeOutputs(sentence, in)

    if (sentOutput.head.valuesIterator.map(_.abs).sum == 0) {
      println(s"${sentence.mkString(" ")}")
    }

    val (prevDelta, mlpGrad, cost) = mlpComputeGradient(sentOutput.last.toDenseVector, label)
    val sentGrad = sentenceComputeGradient(sentence, in, prevDelta, sentOutput)
    (sentGrad ++ mlpGrad, cost)
  }

  protected[mllib] def mlpComputeGradient(
    sentOut: BDV[Double],
    label: BDV[Double]) = {
    if (rand.nextDouble() < 0.001) {
      println(s"sentOut: ${sentOut.valuesIterator.sum / sentOut.length}")
    }
    val mlpIn = BDM.zeros[Double](sentOut.length, 1)
    val mlpLabel = BDM.zeros[Double](label.length, 1)
    mlpIn(::, 0) := sentOut
    mlpLabel(::, 0) := label
    val (mlpOuts, mlpDeltas) = mlp.computeDelta(mlpIn, mlpLabel)
    val mlpGrads = mlp.computeGradientGivenDelta(mlpIn, mlpOuts, mlpDeltas)
    val cost = NNUtil.crossEntropy(mlpOuts.last, mlpLabel)
    val prevDelta: BDM[Double] = mlp.innerLayers.head.weight.t * mlpDeltas.head
    val inputDelta = prevDelta(::, 0)
    if (rand.nextDouble() < 0.001) {
      println(s"mlpDelta: ${inputDelta.valuesIterator.sum / vectorSize}")
    }
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

object SentenceClassifier {
  def train[S <: Iterable[String]](
    dataset: RDD[S],
    word2VecModel: Word2VecModel,
    numIter: Int,
    learningRate: Double,
    fraction: Double): (SentenceClassifier, BDV[Double], Map[String, Int], Map[String, Int]) = {
    val wordVectors = word2VecModel.getVectors
    val vectorSize = wordVectors.head._2.size
    val wordIndex = wordVectors.keys.zipWithIndex.toMap

    val labelIndex = dataset.map(_.head).distinct().collect().zipWithIndex.toMap
    println("label size:" + labelIndex.size)
    val sentences = dataset.map(t => {
      (t.head, t.tail.filter(w => wordIndex.contains(w)))
    }).filter(_._2.size > 4).map(w => {
      val vec = BDV.zeros[Double](labelIndex.size)
      vec :+= 0.1 / vec.length
      vec(labelIndex(w._1)) += 0.9
      (w._2.map(t => wordIndex(t)).toArray, vec)
    })

    val wordVec = BDV.zeros[Double](vectorSize * wordVectors.size)
    wordVectors.foreach { case (word, f) =>
      val offset = wordIndex(word) * vectorSize
      for (i <- 0 until f.length) {
        wordVec(offset + i) = f(i).toDouble
      }
    }

    val sentenceLayer: Array[BaseLayer] = new Array[BaseLayer](6)
    sentenceLayer(0) = new TanhSentenceInputLayer(84, 7, vectorSize)
    sentenceLayer(1) = new DynamicKMaxSentencePooling(7, 0.5)
    // sentenceLayer(1) = new DynamicMaxSentencePooling(6, 0.5)

    val layer2 = new TanhSentenceLayer(84, 128, 5)
    if (layer2.outChannels > 1 && layer2.inChannels > 1) {
      val s = (layer2.outChannels * 0.8).floor.toInt
      for (i <- 0 until layer2.inChannels) {
        for (j <- 0 until s) {
          val offset = (i + j) % layer2.outChannels
          layer2.connTable(i, offset) = 0.0
        }
      }
    }
    sentenceLayer(2) = layer2
    sentenceLayer(3) = new DynamicKMaxSentencePooling(6, 0.5)

    val layer3 = new TanhSentenceLayer(128, 256, 5)
    if (layer3.outChannels > 1 && layer3.inChannels > 1) {
      val s = (layer3.outChannels * 0.7).floor.toInt
      for (i <- 0 until layer3.inChannels) {
        for (j <- 0 until s) {
          val offset = (i + j) % layer3.outChannels
          layer3.connTable(i, offset) = 0.0
        }
      }
    }
    sentenceLayer(4) = layer3
    sentenceLayer(5) = new KMaxSentencePooling(2)
    // sentenceLayer(3) = new MaxSentencePooling(4)

    val mlpLayer = initMLPLayers(Array(256 * 2, 512, labelIndex.size))
    val mlp = new MLP(mlpLayer, Array(0.5, 0.0))
    val sent2vec = new SentenceClassifier(sentenceLayer, mlp, vectorSize)

    val wordBroadcast = dataset.context.broadcast(wordVec)
    val gradientSum = new Array[(BDM[Double], BDV[Double])](sent2vec.numLayer)
    for (iter <- 0 until numIter) {
      val sentBroadcast = dataset.context.broadcast(sent2vec)
      val (grad, loss, miniBatchSize) = trainOnce(sentences,
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
        grad.filter(t => t != null).foreach(m => {
          m._1 :/= miniBatchSize.toDouble
          m._2 :/= miniBatchSize.toDouble
        })
        println(s"loss $iter : " + (loss / miniBatchSize))
        updateParameters(gradientSum, grad, sent2vec, iter, learningRate)
      }
      sentBroadcast.destroy()
    }
    (sent2vec, wordVec, wordIndex, labelIndex)
  }

  // AdaGrad
  def updateParameters(
    etaSum: Array[(BDM[Double], BDV[Double])],
    grad: Array[(BDM[Double], BDV[Double])],
    sent2Vec: SentenceClassifier,
    iter: Int,
    learningRate: Double,
    rho: Double = 1.0,
    epsilon: Double = 1e-3): Unit = {
    val lr = math.min(iter / 11.0, 1) * learningRate
    val numSentenceLayer = sent2Vec.numSentenceLayer

    for (i <- 0 until etaSum.length) {
      if (grad(i) != null) {
        val g2 = grad(i)._1 :* grad(i)._1
        val b2 = grad(i)._2 :* grad(i)._2
        if (etaSum(i) == null) {
          etaSum(i) = (g2, b2)
        } else {
          if (rho > 0D && rho < 1D) {
            etaSum(i)._1 :*= rho
            etaSum(i)._2 :*= rho
          }
          etaSum(i)._1 :+= g2
          etaSum(i)._2 :+= b2
        }
      }
    }
    for (i <- 0 until etaSum.length) {
      if (etaSum(i) != null) {
        val w = grad(i)._1
        val b = grad(i)._2
        val dw = etaSum(i)._1
        val db = etaSum(i)._2
        for (gi <- 0 until w.rows) {
          for (gj <- 0 until w.cols) {
            w(gi, gj) /= (epsilon + math.sqrt(dw(gi, gj)))
          }
        }
        for (gi <- 0 until b.length) {
          b(gi) /= (epsilon + math.sqrt(db(gi)))
        }
      }
    }

    for (i <- 0 until numSentenceLayer) {
      if (grad(i) != null) {
        val layer = sent2Vec.sentenceLayer(i).asInstanceOf[PartialConnectedLayer]
        brzAxpy(-lr, grad(i)._1, layer.weight)
        brzAxpy(-lr, grad(i)._2, layer.bias)
      }
    }

    for (i <- 0 until sent2Vec.mlp.numLayer) {
      val layer = sent2Vec.mlp.innerLayers(i)
      brzAxpy(-lr, grad(numSentenceLayer + i)._1, layer.weight)
      brzAxpy(-lr, grad(numSentenceLayer + i)._2, layer.bias)
    }

  }

  def initMLPLayers(topology: Array[Int]): Array[Layer] = {
    val numLayer = topology.length - 1
    val layers = new Array[Layer](numLayer)
    for (layer <- (0 until numLayer).reverse) {
      val numIn = topology(layer)
      val numOut = topology(layer + 1)
      layers(layer) = if (layer == numLayer - 1) {
        new SoftMaxLayer(numIn, numOut)
      }
      else {
        new TanhLayer(numIn, numOut)
      }
      println(s"layers($layer) = ${numIn} * ${numOut}")
    }
    layers
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
    dataset: RDD[(Array[Int], BDV[Double])],
    sent2Vec: Broadcast[SentenceClassifier],
    word2Vec: Broadcast[BDV[Double]],
    iter: Int,
    fraction: Double): (Array[(BDM[Double], BDV[Double])], Double, Long) = {
    dataset.context.broadcast()
    val numLayer = sent2Vec.value.numLayer
    dataset.sample(false, fraction, 7 * iter + 41).treeAggregate((new Array[(BDM[Double],
      BDV[Double])](numLayer), 0D, 0L))(seqOp = (c, v) => {
      sent2Vec.value.setWord2Vec(word2Vec.value)
      val g = sent2Vec.value.computeGradient(v._1, v._2)
      mergerParameters(c._1, g._1)
      (c._1, c._2 + g._2, c._3 + 1)
    }, combOp = (c1, c2) => {
      mergerParameters(c1._1, c2._1)
      (c1._1, c1._2 + c2._2, c1._3 + c2._3)
    })
  }
}
