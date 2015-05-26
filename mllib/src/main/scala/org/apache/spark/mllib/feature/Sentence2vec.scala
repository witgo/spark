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

import breeze.linalg.{Axis => brzAxis, DenseMatrix => BDM, DenseVector => BDV,
Matrix => BM, Vector => BV, axpy => brzAxpy, max => brzMax, norm => brzNorm, sum => brzSum}
import breeze.stats.distributions.Rand
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.feature.Sentence2vec.ParamInterval
import org.apache.spark.mllib.neuralNetwork.{Layer, TanhLayer, MLP}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

@Experimental
class Sentence2vec(
  val sentenceLayer: Array[BaseLayer],
  val mlp: MLP,
  val vectorSize: Int) extends Serializable with Logging {

  @transient var word2Vec: BDV[Double] = null
  @transient var aliasTable: Sentence2vec.Table = null
  @transient private lazy val numLayer = sentenceLayer.length + mlp.numLayer
  @transient private lazy val numSentenceLayer = sentenceLayer.length
  @transient private lazy val numMLPLayer = mlp.numLayer
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
    val sentOut = input.toDenseVector
    val x = BDM.zeros[Double](sentOut.length, 1)
    x(::, 0) := sentOut
    mlp.predict(x).toDenseVector
  }

  protected def computeGradient(sentence: Array[Int], param: ParamInterval): ParamInterval = {
    val in = sentenceToInput(sentence)
    val sentOutput = sentenceComputeOutputs(sentence, in)
    val sentOut = sentOutput.last.toDenseVector
    cbowWithNegativeSampling(sentence, sentOut, in, sentOutput, param)
    param
  }

  def computeGradientOnce(
    label: Int,
    word: Int,
    sentence: Array[Int],
    sentIn: BDM[Double],
    sentOut: BDV[Double],
    sentOutput: Array[BDM[Double]],
    mlpOutput: Array[BDM[Double]],
    mlpDropOutMasks: Array[BDM[Double]],
    param: ParamInterval): Double = {
    val mlpOut = mlpOutput.last.toDenseVector
    val wordVec = wordToVector(word)
    val margin = -1.0 * mlpOut.dot(wordVec)
    val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
    val topDelta = wordVec :* multiplier

    val (sentTopDelta, mlpGrad) = mlpComputeGradient(sentOut, topDelta, mlpOutput, mlpDropOutMasks)
    val sentGrad = sentenceComputeGradient(sentence, sentIn, sentTopDelta, sentOutput)
    Sentence2vec.mergerParam(param.layerParam, sentGrad ++ mlpGrad)

    if (param.wordParam == null) param.wordParam = BDV.zeros[Double](word2Vec.length)
    brzAxpy(multiplier, mlpOut, wordToVector(param.wordParam, word))
    val loss = if (label > 0) {
      MLUtils.log1pExp(margin)
    } else {
      MLUtils.log1pExp(margin) - margin
    }
    param.loss += loss
    param.miniBatchSize += 1
    loss
  }

  // CBOW with negative-sampling
  protected[mllib] def cbowWithNegativeSampling(
    sentence: Array[Int],
    sentOut: BDV[Double],
    sentIn: BDM[Double],
    sentOutput: Array[BDM[Double]],
    param: ParamInterval): Unit = {
    val k = 5
    val sentenceSize = sentence.size
    val randomize = sentence
    // var randomize = new Array[Int](sentenceSize)
    // Array.copy(sentence, 0, randomize, 0, sentenceSize)
    // randomize = Utils.randomizeInPlace(randomize, rand).slice(0, 3)
    var sampleSize = 0.0
    var loss = 0.0
    for (word <- randomize) {
      val (mlpOutput, mlpDropOutMasks) = mlpComputeOutputs(sentOut)
      loss += computeGradientOnce(1, word, sentence, sentIn, sentOut, sentOutput, mlpOutput,
        mlpDropOutMasks, param)
      sampleSize += 1
      for (i <- 0 until k) {
        val negWord = negativeSampling(sentence)
        assert(negWord != word)
        val (mlpOutput, mlpDropOutMasks) = mlpComputeOutputs(sentOut)
        loss += computeGradientOnce(0, negWord, sentence, sentIn, sentOut, sentOutput, mlpOutput,
          mlpDropOutMasks, param)
        sampleSize += 1
      }
    }
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

  protected[mllib] def mlpComputeOutputs(
    sentOut: BDV[Double]): (Array[BDM[Double]], Array[BDM[Double]]) = {
    val x = BDM.zeros[Double](sentOut.length, 1)
    x(::, 0) := sentOut
    val batchSize = x.cols
    val out = new Array[BDM[Double]](numMLPLayer)
    val dropOutMasks: Array[BDM[Double]] = mlp.dropOutMask(batchSize)
    for (layer <- 0 until numMLPLayer) {
      val output = mlp.innerLayers(layer).forward(if (layer == 0) x else out(layer - 1))
      if (dropOutMasks(layer) != null) {
        assert(output.rows == dropOutMasks(layer).rows)
        output :*= dropOutMasks(layer)
      }
      out(layer) = output
    }

    (out, dropOutMasks)
  }

  protected[mllib] def mlpComputeGradient(
    sentOut: BDV[Double],
    topDelta: BDV[Double],
    out: Array[BDM[Double]],
    dropOutMasks: Array[BDM[Double]]): (BDV[Double], Array[(BDM[Double], BDV[Double])]) = {
    val x = BDM.zeros[Double](sentOut.length, 1)
    x(::, 0) := sentOut
    val delta = new Array[BDM[Double]](numMLPLayer)
    for (i <- (0 until numMLPLayer).reverse) {
      val output = out(i)
      val currentLayer = mlp.innerLayers(i)
      delta(i) = if (i == numMLPLayer - 1) {
        val d = BDM.zeros[Double](topDelta.length, 1)
        d(::, 0) := topDelta
        currentLayer.computeNeuronPrimitive(d, output)
        d
      } else {
        val nextLayer = mlp.innerLayers(i + 1)
        val nextDelta = delta(i + 1)
        nextLayer.previousError(output, currentLayer, nextDelta)
      }
      if (dropOutMasks(i) != null) {
        delta(i) :*= dropOutMasks(i)
      }
    }

    val mlpGrad = mlp.computeGradientGivenDelta(x, out, delta)
    val prevDelta: BDM[Double] = mlp.innerLayers.head.weight.t * delta.head
    val inputDelta = brzSum(prevDelta, brzAxis._1)
    (inputDelta, mlpGrad)
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

  private[mllib] case class ParamInterval(
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

  private[mllib] type Table = (Array[Int], Array[Int], Array[Double])

  private[mllib] def generateAlias(sv: BV[Double], sum: Double): Table = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, used, sum)
  }

  private[mllib] def generateAlias(
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

  private[mllib] def sampleAlias(gen: Random, table: Table): Int = {
    val l = table._1.length
    val bin = gen.nextInt(l)
    val p = table._3(bin)
    if (l * p > gen.nextDouble()) {
      table._1(bin)
    } else {
      table._2(bin)
    }
  }

  private[mllib] def termTable(dataSet: RDD[Array[Int]]): Table = {
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
    val word2Index = dataSet.flatMap { t =>
      t.toSeq.map(w => (w, 1D))
    }.reduceByKey(_ + _).filter(_._2 > minCount).map(_._1).collect().zipWithIndex.toMap
    val sentences = dataSet.map(_.filter(w => word2Index.contains(w))).filter(_.size > 4).
      map(w => w.map(t => word2Index(t)).toArray).persist(StorageLevel.MEMORY_AND_DISK)
    val (sent2vec, word2Vec) = trainSentences(sentences, vectorSize,
      numIter, learningRate, fraction)
    sentences.unpersist(blocking = false)
    (sent2vec, word2Vec, word2Index)
  }

  def trainSentences(
    sentences: RDD[Array[Int]],
    vectorSize: Int,
    numIter: Int,
    learningRate: Double,
    fraction: Double): (Sentence2vec, BDV[Double]) = {
    val sc = sentences.context
    val termSize = sentences.map(_.max).max + 1
    val word2Vec = BDV.rand[Double](vectorSize * termSize, Rand.gaussian)
    word2Vec :*= 1e-2

    val sentenceLayer = initSentenceLayer(vectorSize, Array(84), Array(6))
    val mlpLayer = initMLPLayers(Array(84 * 6, 512, vectorSize), vectorSize)
    val mlp = new MLP(mlpLayer, Array(0.5, 0.0))
    val sent2vec = new Sentence2vec(sentenceLayer, mlp, vectorSize)
    val gradSum = new Array[(BDM[Double], BDV[Double])](sent2vec.numLayer)
    val gradMomentumSum = new Array[(BDM[Double], BDV[Double])](sent2vec.numLayer)
    val wordGradSum: BDV[Double] = BDV.zeros[Double](word2Vec.length)
    val wordMomentumSum: BDV[Double] = BDV.zeros[Double](word2Vec.length)
    val aliasTableBroadcast = sc.broadcast(termTable(sentences))

    for (iter <- 1 to numIter) {
      val word2VecBroadcast = sc.broadcast(word2Vec)
      val sentBroadcast = sc.broadcast(sent2vec)
      val ParamInterval(grad, wordGrad, miniBatchSize, loss) = trainOnce(sentences,
        sentBroadcast, word2VecBroadcast, aliasTableBroadcast, iter, fraction)

      if (Utils.random.nextDouble() < 1e-1) {
        println(s"word2Vec: " + word2Vec.valuesIterator.map(_.abs).sum / word2Vec.size)
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
        adaGradUpdater(gradMomentumSum, gradSum, grad, wordMomentumSum, wordGradSum, wordGrad,
          word2Vec, sent2vec, iter, learningRate)
      }
      sentBroadcast.destroy(blocking = false)
      word2VecBroadcast.destroy(blocking = false)
    }
    (sent2vec, word2Vec)
  }

  def initSentenceLayer(
    vectorSize: Int,
    outChannels: Array[Int],
    kernels: Array[Int]): Array[BaseLayer] = {
    val numLayer = outChannels.length * 2
    val sentenceLayer: Array[BaseLayer] = new Array[BaseLayer](numLayer)
    outChannels.indices.foreach { i =>
      val layerOutChannels = outChannels(i)
      val layerKernels = kernels(i)
      val layer = if (i == 0) {
        new TanhSentenceInputLayer(layerOutChannels, layerKernels, vectorSize)
      } else {
        new TanhSentenceLayer(outChannels(i - 1), layerOutChannels, layerKernels)
      }
      if (i == 2 && layer.outChannels > 1 && layer.inChannels > 1) {
        val s = (layer.outChannels * 0.5).floor.toInt
        for (i <- 0 until layer.inChannels) {
          for (j <- 0 until s) {
            val offset = (i + j) % layer.outChannels
            layer.connTable(i, offset) = 0.0
          }
        }
      }
      sentenceLayer(i * 2) = layer
      val poolingLayer = if (i == outChannels.length - 1) {
        new KMaxSentencePooling(layerKernels)
      } else {
        new DynamicKMaxSentencePooling(layerKernels - 1, 0.5)
      }
      sentenceLayer(i * 2 + 1) = poolingLayer
    }
    sentenceLayer
  }

  def initMLPLayers(topology: Array[Int], vectorSize: Int): Array[Layer] = {
    require(topology.last == vectorSize,
      s"The dimensions of the last payer output must be equal to $vectorSize")
    val numLayer = topology.length - 1
    val layers = new Array[Layer](numLayer)
    for (layer <- (0 until numLayer).reverse) {
      val numIn = topology(layer)
      val numOut = topology(layer + 1)
      layers(layer) = if (layer == numLayer - 1) {
        new TanhLayer(numIn, numOut)
      }
      else {
        new TanhLayer(numIn, numOut)
      }
      println(s"layers($layer) = $numIn * $numOut")
    }
    layers
  }

  // AdaGrad
  def adaGradUpdater(
    gradMomentumSum: Array[(BDM[Double], BDV[Double])],
    gradSum: Array[(BDM[Double], BDV[Double])],
    grad: Array[(BDM[Double], BDV[Double])],
    wordMomentumSum: BDV[Double],
    wordGradSum: BDV[Double],
    wordGrad: BDV[Double],
    word2Vec: BDV[Double],
    sent2Vec: Sentence2vec,
    iter: Int,
    learningRate: Double,
    momentum: Double = 0.9,
    rho: Double = 1.0,
    epsilon: Double = 1e-6): Unit = {
    val lr = learningRate
    val numSentenceLayer = sent2Vec.numSentenceLayer

    if (momentum > 0D && momentum < 1D) {
      brzAxpy(momentum, wordMomentumSum, wordGrad)
      wordMomentumSum := wordGrad
    }
    val wg2 = wordGrad :* wordGrad
    wordGradSum :+= wg2
    if (rho > 0D && rho < 1D) {
      wg2 :*= (1 - rho)
      wordGradSum :*= rho
    }
    for (gi <- 0 until wordGrad.length) {
      wordGrad(gi) /= (epsilon + math.sqrt(wordGradSum(gi)))
    }
    brzAxpy(-lr, wordGrad, word2Vec)

    if (momentum > 0D && momentum < 1D) {
      for (i <- 0 until gradSum.length) {
        if (grad(i) != null) {
          if (gradMomentumSum(i) == null) {
            gradMomentumSum(i) = (BDM.zeros[Double](grad(i)._1.rows, grad(i)._1.cols),
              BDV.zeros[Double](grad(i)._2.length))
          }
          val mw = gradMomentumSum(i)._1
          val mb = gradMomentumSum(i)._2
          val gw = grad(i)._1
          val gb = grad(i)._2
          mw :*= momentum
          mw :+= gw
          gw := mw
          brzAxpy(momentum, mb, gb)
          mb := gb
        }
      }
    }

    for (i <- 0 until gradSum.length) {
      if (grad(i) != null) {
        val g2 = grad(i)._1 :* grad(i)._1
        val b2 = grad(i)._2 :* grad(i)._2
        if (gradSum(i) == null) {
          gradSum(i) = (g2, b2)
        } else {
          if (rho > 0D && rho < 1D) {
            g2 :*= (1D - rho)
            b2 :*= (1D - rho)
            gradSum(i)._1 :*= rho
            gradSum(i)._2 :*= rho
          }
          gradSum(i)._1 :+= g2
          gradSum(i)._2 :+= b2
        }
      }
    }
    for (i <- 0 until gradSum.length) {
      if (gradSum(i) != null) {
        val w = grad(i)._1
        val b = grad(i)._2
        val dw = gradSum(i)._1
        val db = gradSum(i)._2
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

  // Equilibrated SGD
  def equilibratedUpdater(
    gradMomentumSum: Array[(BDM[Double], BDV[Double])],
    gradSum: Array[(BDM[Double], BDV[Double])],
    grad: Array[(BDM[Double], BDV[Double])],
    wordMomentumSum: BDV[Double],
    wordGradSum: BDV[Double],
    wordGrad: BDV[Double],
    word2Vec: BDV[Double],
    sent2Vec: Sentence2vec,
    iter: Int,
    learningRate: Double,
    momentum: Double = 0,
    epsilon: Double = 1e-6): Unit = {
    val lr = learningRate
    val numSentenceLayer = sent2Vec.numSentenceLayer

    //  val wg2 = (wordGrad :* wordGrad).mapValues(v => v * math.pow(Utils.random.nextGaussian(), 2))
    //  wordGradSum :+= wg2
    //  for (gi <- 0 until wordGrad.length) {
    //    wordGrad(gi) /= (epsilon + math.sqrt(wordGradSum(gi) / iter))
    //  }

    val wg2 = (wordGrad :* wordGrad)
    wordGradSum :+= wg2
    for (gi <- 0 until wordGrad.length) {
      wordGrad(gi) /= (epsilon + math.sqrt(wordGradSum(gi)))
    }

    if (momentum > 0D && momentum < 1D) {
      wordGrad :*= (1 - momentum)
      brzAxpy(momentum, wordMomentumSum, wordGrad)
      wordMomentumSum := wordGrad
    }
    brzAxpy(-lr, wordGrad, word2Vec)

    for (i <- 0 until gradSum.length) {
      if (grad(i) != null) {
        val g2 = (grad(i)._1 :* grad(i)._1).mapValues(v =>
          v * math.pow(Utils.random.nextGaussian(), 2))
        val b2 = (grad(i)._2 :* grad(i)._2).mapValues(v =>
          v * math.pow(Utils.random.nextGaussian(), 2))
        if (gradSum(i) == null) {
          gradSum(i) = (g2, b2)
        } else {
          gradSum(i)._1 :+= g2
          gradSum(i)._2 :+= b2
        }
      }
    }
    for (i <- 0 until gradSum.length) {
      if (gradSum(i) != null) {
        val w = grad(i)._1
        val b = grad(i)._2
        val dw = gradSum(i)._1
        val db = gradSum(i)._2
        for (gi <- 0 until w.rows) {
          for (gj <- 0 until w.cols) {
            w(gi, gj) /= (epsilon + math.sqrt(dw(gi, gj) / iter))
          }
        }
        for (gi <- 0 until b.length) {
          b(gi) /= (epsilon + math.sqrt(db(gi) / iter))
        }
      }
    }

    if (momentum > 0D && momentum < 1D) {
      for (i <- 0 until gradSum.length) {
        if (grad(i) != null) {
          if (gradMomentumSum(i) == null) {
            gradMomentumSum(i) = (BDM.zeros[Double](grad(i)._1.rows, grad(i)._1.cols),
              BDV.zeros[Double](grad(i)._2.length))
          }
          val mw = gradMomentumSum(i)._1
          val mb = gradMomentumSum(i)._2
          val gw = grad(i)._1
          val gb = grad(i)._2
          gw :*= (1 - momentum)
          mw :*= momentum
          mw :+= gw
          gw := mw

          gb :*= (1 - momentum)
          brzAxpy(momentum, mb, gb)
          mb := gb
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

  private[mllib] def mergerParam(
    a: Array[(BDM[Double], BDV[Double])],
    b: Array[(BDM[Double], BDV[Double])]): Unit = {
    for (i <- 0 until a.length) {
      if (a(i) == null) {
        a(i) = b(i)
      } else if (b(i) != null) {
        a(i)._1 :+= b(i)._1
        a(i)._2 :+= b(i)._2
      }
    }
  }

  private[mllib] def trainOnce(
    dataSet: RDD[Array[Int]],
    sent2Vec: Broadcast[Sentence2vec],
    word2Vec: Broadcast[BDV[Double]],
    aliasTableBroadcast: Broadcast[Table],
    iter: Int,
    fraction: Double): ParamInterval = {
    dataSet.context.broadcast()
    val numLayer = sent2Vec.value.numLayer
    val zeroValue = ParamInterval(new Array[(BDM[Double], BDV[Double])](numLayer), null, 0, 0)
    dataSet.sample(false, fraction).treeAggregate(zeroValue)(seqOp = (c, v) => {
      val s = sent2Vec.value
      s.setSeed(Utils.random.nextLong())
      s.setWord2Vec(word2Vec.value).setAliasTable(aliasTableBroadcast.value)
      s.computeGradient(v, c)
    }, combOp = (c1, c2) => {
      c1.mergerParam(c2)
    })
  }
}
