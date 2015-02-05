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

package org.apache.spark.mllib.clustering

import java.util.Random

import scala.reflect.ClassTag

import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV,
sum => brzSum, norm => brzNorm}

import org.apache.spark.mllib.linalg.{Vectors, DenseVector => SDV, SparseVector => SSV}

class LDAModel private[mllib](
  private[mllib] val gtc: BDV[Double],
  private[mllib] val ttc: Array[BSV[Double]],
  val alpha: Double,
  val beta: Double) extends Serializable {

  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta)
  }

  val (numTopics, numTerms) = (gtc.size, ttc.size)
  lazy val numTokens = brzSum(gtc)

  def globalTopicCounter = Vectors.fromBreeze(gtc)

  def topicTermCounter = ttc.map(t => Vectors.fromBreeze(t))

  def inference(doc: SSV, totalIter: Int = 10, burnIn: Int = 5, rand: Random = new Random): SSV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")

    val topicDist = BSV.zeros[Double](numTopics)
    val bDoc = new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size)
    val tokens = vector2Array(bDoc)
    val topics = new Array[Int](tokens.length)
    var docTopicCounter = uniformDistSampler(tokens, topics, rand)
    for (i <- 0 until totalIter) {
      docTopicCounter = sampleTokens(docTopicCounter, tokens, topics, rand)
      if (i + burnIn >= totalIter) topicDist :+= docTopicCounter
    }
    topicDist.compact()
    topicDist :/= brzNorm(topicDist, 1)
    Vectors.fromBreeze(topicDist).asInstanceOf[SSV]
  }

  private[mllib] def vector2Array(vec: BV[Int]): Array[Int] = {
    val docLen = brzSum(vec)
    var offset = 0
    val sent = new Array[Int](docLen)
    vec.activeIterator.foreach { case (term, cn) =>
      for (i <- 0 until cn) {
        sent(offset) = term
        offset += 1
      }
    }
    sent
  }

  private[mllib] def sampleTokens(
    docTopicCounter: BSV[Double],
    tokens: Array[Int],
    topics: Array[Int],
    rand: Random): BSV[Double] = {
    var d: BSV[Double] = null
    for (i <- 0 until topics.length) {
      val term = tokens(i)
      val currentTopic = topics(i)
      val docProposal = rand.nextDouble() < 0.5
      val proposalTopic = if (docProposal) {
        gibbsSamplerWord(rand, t, w(term))
      } else {
        if (d == null) d = this.d(docTopicCounter, alpha)
        gibbsSamplerDoc(rand, d, alpha, currentTopic)
      }
      val newTopic = metropolisHastingsSampler(rand, docTopicCounter, ttc(term),
        gtc, beta, alpha, alpha, numTokens, numTerms, currentTopic,
        proposalTopic, docProposal)

      if (newTopic != currentTopic) {
        docTopicCounter(currentTopic) -= 1D
        docTopicCounter(newTopic) += 1D
        if (docTopicCounter(currentTopic) == 0) {
          docTopicCounter.compact()
        }
        d = null
      }
      topics(i) = newTopic
    }
    docTopicCounter
  }

  private[mllib] def uniformDistSampler(
    tokens: Array[Int],
    topics: Array[Int],
    rand: Random): BSV[Double] = {
    val docTopicCounter = BSV.zeros[Double](numTopics)
    for (i <- 0 until tokens.length) {
      val topic = LDAUtils.uniformDistSampler(rand, numTopics)
      topics(i) = topic
      docTopicCounter(topic) += 1D
    }
    docTopicCounter
  }

  @inline private def gibbsSamplerWord(
    rand: Random,
    t: BDV[Double],
    w: BSV[Double]): Int = {
    val distSum = rand.nextDouble * (t(numTopics - 1) + w.data(w.used - 1))
    val fun = indexWord(t, w) _
    val topic = LDAUtils.binarySearchInterval(fun, distSum, 0, numTopics, true)
    math.min(topic, numTopics - 1)
  }

  private def gibbsSamplerDoc(
    rand: Random,
    d: BSV[Double],
    alpha: Double,
    currentTopic: Int): Int = {
    val numTopics = d.length
    val adjustment = -1D
    val lastSum = d.data(d.used - 1) + alpha * numTopics + adjustment
    val distSum = rand.nextDouble() * lastSum
    val fun = (topic: Int) => {
      val lastSum = LDAUtils.binarySearchSparseVector(topic, d)
      val tSum = alpha * (topic + 1)
      if (topic >= currentTopic) lastSum + tSum + adjustment else lastSum + tSum
    }
    val topic = LDAUtils.binarySearchInterval(fun, distSum, 0, numTopics, true)
    math.min(topic, numTopics - 1)
  }

  @inline private def indexWord(
    t: BDV[Double],
    w: BSV[Double])(topic: Int) = {
    val lastWS = LDAUtils.binarySearchSparseVector(topic, w)
    val lastTS = t(topic)
    lastWS + lastTS
  }

  // scalastyle:off
  def metropolisHastingsSampler(
    rand: Random,
    docTopicCounter: BSV[Double],
    termTopicCounter: BSV[Double],
    totalTopicCounter: BDV[Double],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numTokens: Double,
    numTerms: Double,
    currentTopic: Int,
    newTopic: Int,
    docProposal: Boolean): Int = {
    if (newTopic == currentTopic) return currentTopic

    val ctp = tokenTopicProb(docTopicCounter, termTopicCounter, totalTopicCounter,
      beta, alpha, alphaAS, numTokens, numTerms, currentTopic, true)
    val ntp = tokenTopicProb(docTopicCounter, termTopicCounter, totalTopicCounter,
      beta, alpha, alphaAS, numTokens, numTerms, newTopic, false)
    val cwp = if (docProposal) {
      docTopicProb(docTopicCounter, currentTopic, alpha, true)
    } else {
      wordTopicProb(termTopicCounter, totalTopicCounter, currentTopic,
        numTerms, beta)
    }
    val nwp = if (docProposal) {
      docTopicProb(docTopicCounter, newTopic, alpha, false)
    } else {
      wordTopicProb(termTopicCounter, totalTopicCounter, newTopic,
        numTerms, beta)
    }
    val pi = (ntp * cwp) / (ctp * nwp)

    if (rand.nextDouble() < 1e-7) {
      println(s"Model Pi: ${pi}")
      println(s"($ntp * $cwp) / ($ctp * $nwp) ")
    }

    if (rand.nextDouble() < math.min(1.0, pi)) {
      newTopic
    } else {
      currentTopic
    }
  }

  // scalastyle:on

  @inline private def tokenTopicProb(
    docTopicCounter: BSV[Double],
    termTopicCounter: BSV[Double],
    totalTopicCounter: BDV[Double],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numTokens: Double,
    numTerms: Double,
    topic: Int,
    isAdjustment: Boolean): Double = {
    val ratio = (totalTopicCounter(topic) + alphaAS) / (numTokens + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    val adjustment = if (isAdjustment) -1.0 else 0.0
    (termTopicCounter(topic) + beta) * (docTopicCounter(topic) + adjustment + asPrior) /
      (totalTopicCounter(topic) + (numTerms * beta))
  }

  @inline private def wordTopicProb(
    termTopicCounter: BSV[Double],
    totalTopicCounter: BDV[Double],
    topic: Int,
    numTerms: Double,
    beta: Double): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  @inline private def docTopicProb(
    docTopicCounter: BSV[Double],
    topic: Int,
    alpha: Double,
    isAdjustment: Boolean): Double = {
    if (isAdjustment) {
      docTopicCounter(topic) + alpha - 1
    } else {
      docTopicCounter(topic) + alpha
    }
  }

  @transient private lazy val t = {
    val t = BDV.zeros[Double](numTopics)
    val termSum = numTerms * beta
    var lastSum = 0D
    for (i <- 0 until numTopics) {
      t(i) = beta / (gtc(i) + termSum)
      lastSum = t(i) + lastSum
      t(i) = lastSum
    }
    t
  }

  @transient private lazy val w = {
    val w = new Array[BSV[Double]](numTerms)
    val termSum = numTerms * beta
    for (term <- 0 until numTerms) {
      val bsv = BSV.zeros[Double](numTopics)
      var lastSum = 0D
      ttc(term).activeIterator.foreach { case (topic, cn) =>
        bsv(topic) = cn / (gtc(topic) + termSum)
        lastSum = bsv(topic) + lastSum
        bsv(topic) = lastSum
      }
      bsv.compact()
      w(term) = bsv
    }
    w
  }


  @inline private def d(
    docTopicCounter: BSV[Double],
    alpha: Double): BSV[Double] = {
    val numTopics = docTopicCounter.length
    val used = docTopicCounter.used
    val index = docTopicCounter.index
    val data = docTopicCounter.data
    val d = new Array[Double](used)
    assert(used > 0)
    var lastSum = 0D
    var i = 0

    while (i < used) {
      val topic = index(i)
      val lastW = data(i) // + (topic + 1) * alpha
      lastSum += lastW
      d(i) = lastSum
      i += 1
    }
    new BSV[Double](index, d, used, numTopics)
  }

  private[mllib] def mergeOne(term: Int, topic: Int, inc: Int) = {
    gtc(topic) += inc
    ttc(term)(topic) += inc
    this
  }

  private[mllib] def merge(term: Int, counter: BV[Int]) = {
    counter.activeIterator.foreach { case (topic, cn) =>
      mergeOne(term, topic, cn)
    }
    this
  }

  private[mllib] def merge(other: LDAModel) = {
    gtc :+= other.gtc
    for (i <- 0 until ttc.length) {
      ttc(i) :+= other.ttc(i)
    }
    this
  }

}

object LDAModel {
  def apply(numTopics: Int, numTerms: Int, alpha: Double = 0.1, beta: Double = 0.01) = {
    new LDAModel(
      BDV.zeros[Double](numTopics),
      (0 until numTerms).map(_ => BSV.zeros[Double](numTopics)).toArray, alpha, beta)
  }
}


object LDAUtils {

  /**
   * A uniform distribution sampler
   */
  @inline private[mllib] def uniformDistSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }

  def binarySearchArray[K](
    index: Array[K],
    key: K,
    begin: Int,
    end: Int,
    greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    binarySearchInterval(i => index(i), key, begin, end, greater)
  }

  def binarySearchInterval[K](
    index: Int => K,
    key: K,
    begin: Int,
    end: Int,
    greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = index(mid)
      if (ord.lt(v, key)) {
        b = mid + 1
      }
      else if (ord.gt(v, key)) {
        e = mid - 1
      }
      else {
        return mid
      }
    }
    val v = index(mid)
    mid = if ((greater && ord.gteq(v, key)) || (!greater && ord.lteq(v, key))) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }

    // 测试代码验证 index(mid) 大于等于 index(mid - 1) 小于等于 index(mid + 1)
    //  if (greater) {
    //    if (mid < end) assert(ord.gteq(index(mid), key))
    //    if (mid > 0) assert(ord.lteq(index(mid - 1), key))
    //  } else {
    //    if (mid > 0) assert(ord.lteq(index(mid), key))
    //    if (mid < end - 1) assert(ord.gteq(index(mid + 1), key))
    //  }
    mid
  }

  @inline private[mllib] def binarySearchSparseVector(index: Int, w: BSV[Double]) = {
    val pos = binarySearchArray(w.index, index, 0, w.used, false)
    if (pos > -1) {
      w.data(pos)
    }
    else {
      0D
    }
  }
}