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

import java.util.{PriorityQueue => JPriorityQueue, Random}

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, Matrix => BM,
max => brzMax, Axis => brzAxis, sum => brzSum, axpy => brzAxpy, norm => brzNorm, Vector => BV}
import breeze.stats.distributions.Rand
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.feature.Sentence2vec.ParamInterval
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

@Experimental
class Sentence2vec(
  val sentenceLayer: Array[BaseLayer],
  val vectorSize: Int) extends Serializable with Logging {

  @transient var word2Vec: BDV[Double] = null
  @transient var aliasTable: Sentence2vec.Table = null
  @transient private lazy val numSentenceLayer = sentenceLayer.length
  @transient private lazy val numLayer = numSentenceLayer
  @transient private lazy val rand: Random = new XORShiftRandom()
  @transient private lazy val wordSize: Int = word2Vec.length / vectorSize

  def setSeed(seed: Long): this.type = {
    rand.setSeed(seed)
    this
  }

  def setWord2Vec(word2Vec: BDV[Double]): this.type = {
    this.word2Vec = word2Vec
    this
  }

  def setAliasTable(table: Sentence2vec.Table): this.type = {
    this.aliasTable = table
    this
  }

  def predict(sentence: Array[Int]): BDV[Double] = {
    var input = sentenceToInput(sentence)
    for (i <- 0 until sentenceLayer.length) {
      input = sentenceLayer(i).forward(input)
    }
    input.toDenseVector
  }

  protected def computeGradient(sentence: Array[Int], param: ParamInterval): ParamInterval = {
    val in = sentenceToInput(sentence)
    val sentOutput = sentenceComputeOutputs(sentence, in)
    CBOWNS(sentence, sentOutput.last.toDenseVector, in, sentOutput, param)
    param
  }

  // CBOW with negative-sampling
  protected[mllib] def CBOWNS(
    sentence: Array[Int],
    sentOut: BDV[Double],
    in: BDM[Double],
    output: Array[BDM[Double]],
    param: ParamInterval): Unit = {
    val k = 2
    val sentenceSize = sentence.size
    var randomize = new Array[Int](sentenceSize)
    Array.copy(sentence, 0, randomize, 0, sentenceSize)
    randomize = Utils.randomizeInPlace(randomize, rand).slice(0, 3)
    var sampleSize = 0.0
    var loss = 0.0
    for (word <- randomize) {
      loss += computeGradientOnce(1, word, sentOut, param, sentence, in, output)
      sampleSize += 1
      for (i <- 0 until k) {
        val negWord = negativeSampling(sentence)
        assert(negWord != word)
        loss += computeGradientOnce(0, negWord, sentOut, param, sentence, in, output)
        sampleSize += 1
      }
    }
  }

  def computeGradientOnce(
    label: Int,
    word: Int,
    sentOut: BDV[Double],
    param: ParamInterval,
    sentence: Array[Int],
    in: BDM[Double],
    output: Array[BDM[Double]]): Double = {
    val wordVec = wordToVector(word)
    val margin = -1.0 * sentOut.dot(wordVec)
    val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
    val mlpDelta = wordVec :* multiplier
    val sentGrad = sentenceComputeGradient(sentence, in, mlpDelta, output)
    Sentence2vec.mergerParam(param.layerParam, sentGrad)
    if (param.wordParam == null) param.wordParam = BDV.zeros[Double](word2Vec.length)
    brzAxpy(multiplier, sentOut, wordToVector(param.wordParam, word))
    val loss = if (label > 0) {
      MLUtils.log1pExp(margin)
    } else {
      MLUtils.log1pExp(margin) - margin
    }
    param.loss += loss
    param.miniBatchSize += 1
    loss
  }

  private def negativeSampling(sentence: Array[Int]): Int = {
    var sWord = -1
    do {
      sWord = Sentence2vec.sampleAlias(rand, aliasTable)
    } while (sentence.contains(sWord))
    sWord
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
    in: BDM[Double],
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
      grads(i) = sentenceLayer(i).backward(if (i == 0) in else output(i - 1), delta(i))
    }
    grads
  }

  private def wordToVector(pos: Int): BDV[Double] = {
    wordToVector(word2Vec, pos)
  }

  private def wordToVector(wordVec: BDV[Double], pos: Int): BDV[Double] = {
    wordVec(pos * vectorSize until (pos + 1) * vectorSize)
  }

  private[mllib] def sentenceToInput(sentence: Array[Int]): BDM[Double] = {
    val vectors = sentence.map { s =>
      word2Vec(s * vectorSize until (s + 1) * vectorSize)
    }
    BDV.vertcat(vectors.toArray: _*).asDenseMatrix.t
  }

}

object Sentence2vec {

  case class ParamInterval(
    var layerParam: Array[(BDM[Double], BDV[Double])],
    var wordParam: BDV[Double],
    var miniBatchSize: Long,
    var loss: Double) {

    def mergerParam(other: ParamInterval): this.type = {
      Sentence2vec.mergerParam(layerParam, other.layerParam)
      if (wordParam == null) {
        wordParam = other.wordParam
      } else if (wordParam != null && other.wordParam != null) {
        wordParam :+= other.wordParam
      }
      loss += other.loss
      miniBatchSize += other.miniBatchSize
      this
    }
  }

  @transient private lazy val tableOrdering = new scala.math.Ordering[(Int, Double)] {
    override def compare(x: (Int, Double), y: (Int, Double)): Int = {
      Ordering.Double.compare(x._2, y._2)
    }
  }

  @transient private lazy val tableReverseOrdering = tableOrdering.reverse

  type Table = (Array[Int], Array[Int], Array[Double])

  def generateAlias(sv: BV[Double], sum: Double): Table = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, used, sum)
  }

  def generateAlias(
    probs: Iterator[(Int, Double)],
    used: Int,
    sum: Double): Table = {
    val pMean = 1.0 / used
    val table = (new Array[Int](used), new Array[Int](used), new Array[Double](used))

    val lq = new JPriorityQueue[(Int, Double)](used, tableOrdering)
    val hq = new JPriorityQueue[(Int, Double)](used, tableReverseOrdering)

    probs.slice(0, used).foreach { pair =>
      val i = pair._1
      val pi = pair._2 / sum
      if (pi < pMean) {
        lq.add((i, pi))
      } else {
        hq.add((i, pi))
      }
    }

    var offset = 0
    while (!lq.isEmpty & !hq.isEmpty) {
      val (i, pi) = lq.remove()
      val (h, ph) = hq.remove()
      table._1(offset) = i
      table._2(offset) = h
      table._3(offset) = pi
      val pd = ph - (pMean - pi)
      if (pd >= pMean) {
        hq.add((h, pd))
      } else {
        lq.add((h, pd))
      }
      offset += 1
    }
    while (!hq.isEmpty) {
      val (h, ph) = hq.remove()
      assert(ph - pMean < 1e-8)
      table._1(offset) = h
      table._2(offset) = h
      table._3(offset) = ph
      offset += 1
    }

    while (!lq.isEmpty) {
      val (i, pi) = lq.remove()
      assert(pMean - pi < 1e-8)
      table._1(offset) = i
      table._2(offset) = i
      table._3(offset) = pi
      offset += 1
    }
    table
  }

  def sampleAlias(gen: Random, table: Table): Int = {
    val l = table._1.length
    val bin = gen.nextInt(l)
    val p = table._3(bin)
    if (l * p > gen.nextDouble()) {
      table._1(bin)
    } else {
      table._2(bin)
    }
  }

  def termTable(dataSet: RDD[Array[Int]]): Table = {
    val probs = dataSet.flatMap(t => t.toSeq.map(w => (w, 1D))).reduceByKey(_ + _).
      collect().sortWith((a, b) => a._2 > b._2)
    var sum = 0.0
    probs.foreach(x => sum += x._2)
    generateAlias(probs.toIterator, probs.length, sum)
  }

  def train[S <: Iterable[String]](
    dataSet: RDD[S],
    vectorSize: Int,
    numIter: Int,
    learningRate: Double,
    fraction: Double): (Sentence2vec, BDV[Double], Map[String, Int]) = {
    val minCount = 5
    val word2Index = dataSet.flatMap { t => t.toSeq.map(w => (w, 1D))
    }.reduceByKey(_ + _).filter(_._2 > minCount).map(_._1).collect().zipWithIndex.toMap
    val sentences = dataSet.map(_.filter(w => word2Index.contains(w))).filter(_.size > 4).
      map(w => w.map(t => word2Index(t)).toArray)
    val word2Vec = BDV.rand[Double](vectorSize * word2Index.size, Rand.gaussian)
    word2Vec :*= 1e-2
    val sentenceLayer: Array[BaseLayer] = new Array[BaseLayer](4)
    sentenceLayer(0) = new TanhSentenceInputLayer(64, 7, vectorSize)
    sentenceLayer(1) = new DynamicKMaxSentencePooling(6, 0.5)
    val layer = new TanhSentenceLayer(64, vectorSize / 4, 5)
    if (layer.outChannels > 1 && layer.inChannels > 1) {
      val s = (layer.outChannels * 0.5).floor.toInt
      for (i <- 0 until layer.inChannels) {
        for (j <- 0 until s) {
          val offset = (i + j) % layer.outChannels
          layer.connTable(i, offset) = 0.0
        }
      }
    }
    sentenceLayer(2) = layer
    sentenceLayer(3) = new KMaxSentencePooling(4)
    val sent2vec = new Sentence2vec(sentenceLayer, vectorSize)

    val gradientSum = new Array[(BDM[Double], BDV[Double])](sent2vec.numLayer)
    val wordGradSum: BDV[Double] = BDV.zeros[Double](word2Vec.length)
    val aliasTableBroadcast = dataSet.context.broadcast(termTable(sentences))

    for (iter <- 0 until numIter) {
      val word2VecBroadcast = dataSet.context.broadcast(word2Vec)
      val sentBroadcast = dataSet.context.broadcast(sent2vec)
      val ParamInterval(grad, wordGrad, miniBatchSize, loss) = trainOnce(sentences,
        sentBroadcast, word2VecBroadcast, aliasTableBroadcast, iter, fraction)
      if (Utils.random.nextDouble() < 1e-1) {
        println(s"word2Vec: " + word2Vec.valuesIterator.map(_.abs).sum / word2Vec.size)
        sentenceLayer.zipWithIndex.foreach { case (b, i) =>
          b match {
            case s: SentenceLayer =>
              val weight = s.weight
              println(s"sentenceLayer weight $i: " + weight.valuesIterator.map(_.abs).sum / weight.size)
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
        updateParameters(gradientSum, grad, wordGradSum, wordGrad, word2Vec, sent2vec, iter, learningRate)
      }
      sentBroadcast.destroy()
    }
    (sent2vec, word2Vec, word2Index)
  }

  // AdaGrad
  def updateParameters(
    etaSum: Array[(BDM[Double], BDV[Double])],
    grad: Array[(BDM[Double], BDV[Double])],
    wordGradSum: BDV[Double],
    wordGrad: BDV[Double],
    word2Vec: BDV[Double],
    sent2Vec: Sentence2vec,
    iter: Int,
    learningRate: Double,
    rho: Double = 1.0,
    epsilon: Double = 1e-6): Unit = {
    val lr = learningRate
    val numSentenceLayer = sent2Vec.numSentenceLayer

    if (rho > 0D && rho < 1D) {
      wordGradSum :*= rho
    }
    val wg2 = wordGrad :* wordGrad
    wordGradSum :+= wg2
    for (gi <- 0 until wordGrad.length) {
      wordGrad(gi) /= (epsilon + math.sqrt(wordGradSum(gi)))
    }
    brzAxpy(-lr, wordGrad, word2Vec)

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
  }

  def mergerParam(
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
    aliasTableBroadcast: Broadcast[Table],
    iter: Int,
    fraction: Double): ParamInterval = {
    dataset.context.broadcast()
    val numLayer = sent2Vec.value.numLayer
    val zeroValue = ParamInterval(new Array[(BDM[Double], BDV[Double])](numLayer), null, 0, 0)
    dataset.sample(false, fraction).treeAggregate(zeroValue)(seqOp = (c, v) => {
      val s = sent2Vec.value
      s.setSeed(Utils.random.nextLong())
      s.setWord2Vec(word2Vec.value).setAliasTable(aliasTableBroadcast.value)
      s.computeGradient(v, c)
    }, combOp = (c1, c2) => {
      c1.mergerParam(c2)
    })
  }
}
