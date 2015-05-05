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

import java.lang.ref.SoftReference
import java.util.Random

import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV,
sum => brzSum, norm => brzNorm}

import org.apache.spark.mllib.linalg.{Vectors, DenseVector => SDV, SparseVector => SSV}
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.util.random.XORShiftRandom

import LDAUtils._

class LDAModel private[mllib](
  private[mllib] val gtc: BDV[Double],
  private[mllib] val ttc: Array[BSV[Double]],
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double) extends Serializable {

  @transient private lazy val numTopics = gtc.size
  @transient private lazy val numTerms = ttc.size
  @transient private lazy val numTokens = brzSum(gtc)
  @transient private lazy val betaSum = numTerms * beta
  @transient private lazy val alphaSum = numTopics * alpha
  @transient private lazy val termSum = numTokens + alphaAS * numTopics
  @transient private lazy val wordTableCache =
    new AppendOnlyMap[Int, SoftReference[(Double, AliasTable)]]()
  @transient private lazy val (t, tSum) = {
    val dv = tDense(gtc, numTokens, numTerms, alpha, alphaAS, beta)
    (AliasTable.generateAlias(dv._2, dv._1), dv._1)
  }
  @transient private lazy val rand = new XORShiftRandom()

  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta, alpha)
  }

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  def globalTopicCounter = Vectors.fromBreeze(gtc)

  def topicTermCounter = ttc.map(t => Vectors.fromBreeze(t))

  def inference(
    doc: SSV,
    totalIter: Int = 10,
    burnIn: Int = 5): SSV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")

    val topicDist = BSV.zeros[Double](numTopics)
    val tokens = vector2Array(new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size))
    val topics = new Array[Int](tokens.length)

    var docTopicCounter = uniformDistSampler(tokens, topics)
    for (i <- 0 until totalIter) {
      docTopicCounter = sampleTokens(docTopicCounter, tokens, topics)
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

  private[mllib] def uniformDistSampler(
    tokens: Array[Int],
    topics: Array[Int]): BSV[Double] = {
    val docTopicCounter = BSV.zeros[Double](numTopics)
    for (i <- 0 until tokens.length) {
      val topic = uniformSampler(rand, numTopics)
      topics(i) = topic
      docTopicCounter(topic) += 1D
    }
    docTopicCounter
  }

  private[mllib] def sampleTokens(
    docTopicCounter: BSV[Double],
    tokens: Array[Int],
    topics: Array[Int]): BSV[Double] = {
    for (i <- 0 until topics.length) {
      val termId = tokens(i)
      val currentTopic = topics(i)
      val d = dSparse(gtc, ttc(termId), docTopicCounter,
        currentTopic, numTokens, numTerms, alpha, alphaAS, beta)

      val (wSum, w) = wordTable(wordTableCache, gtc, ttc(termId), termId,
        numTokens, numTerms, alpha, alphaAS, beta)
      val newTopic = tokenSampling(rand, t, tSum, w, wSum, d)
      if (newTopic != currentTopic) {
        docTopicCounter(newTopic) += 1D
        docTopicCounter(currentTopic) -= 1D
        topics(i) = newTopic
        if (docTopicCounter(currentTopic) == 0) {
          docTopicCounter.compact()
        }
      }
    }
    docTopicCounter
  }

  private def tokenSampling(
    gen: Random,
    t: AliasTable,
    tSum: Double,
    w: AliasTable,
    wSum: Double,
    d: BSV[Double]): Int = {
    val index = d.index
    val data = d.data
    val used = d.used
    val dSum = data(d.used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextDouble() * dSum
      val pos = binarySearchInterval(data, dGenSum, 0, used, true)
      index(pos)
    } else if (genSum < (dSum + wSum)) {
      w.sampleAlias(gen)
    } else {
      t.sampleAlias(gen)
    }
  }

  private def dSparse(
    totalTopicCounter: BDV[Double],
    termTopicCounter: BSV[Double],
    docTopicCounter: BSV[Double],
    currentTopic: Int,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): BSV[Double] = {
    val numTopics = totalTopicCounter.length
    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    val d = BSV.zeros[Double](numTopics)
    var sum = 0.0
    docTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = if (currentTopic == topic && t._2 != 1) t._2 - 1 else t._2
      // val last = count * termSum * (termTopicCounter(topic) + beta) /
      //  ((totalTopicCounter(topic) + betaSum) * termSum)
      val last = count * (termTopicCounter(topic) + beta) /
        (totalTopicCounter(topic) + betaSum)
      sum += last
      d(topic) = sum
    }
    d
  }

  private def wordTable(
    cacheMap: AppendOnlyMap[Int, SoftReference[(Double, AliasTable)]],
    totalTopicCounter: BDV[Double],
    termTopicCounter: BSV[Double],
    termId: Int,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, AliasTable) = {
    if (termTopicCounter.used == 0) return (0.0, null)
    var w = cacheMap(termId)
    if (w == null || w.get() == null) {
      val t = wSparse(totalTopicCounter, termTopicCounter,
        numTokens, numTerms, alpha, alphaAS, beta)
      w = new SoftReference((t._1, AliasTable.generateAlias(t._2, t._1)))
      cacheMap.update(termId, w)

    }
    w.get()
  }

  private def wSparse(
    totalTopicCounter: BDV[Double],
    termTopicCounter: BSV[Double],
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BSV[Double]) = {
    val w = BSV.zeros[Double](numTopics)
    var sum = 0.0
    termTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      val last = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      w(topic) = last
      sum += last
    }
    (sum, w)
  }

  private def tDense(
    totalTopicCounter: BDV[Double],
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BDV[Double]) = {
    val t = BDV.zeros[Double](numTopics)
    var sum = 0.0
    for (topic <- 0 until numTopics) {
      val last = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  private[mllib] def merge(term: Int, counter: BV[Int]) = {
    counter.activeIterator.foreach { case (topic, cn) =>
      mergeOne(term, topic, cn)
    }
    this
  }

  private[mllib] def mergeOne(term: Int, topic: Int, inc: Int) = {
    gtc(topic) += inc
    ttc(term)(topic) += inc
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
      (0 until numTerms).map(_ => BSV.zeros[Double](numTopics)).toArray, alpha, beta, alpha)
  }
}

private[mllib] object LDAUtils {

  def uniformSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }

  def binarySearchInterval(
    index: Array[Double],
    key: Double,
    begin: Int,
    end: Int,
    greater: Boolean): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = index(mid)
      if (v < key) {
        b = mid + 1
      }
      else if (v > key) {
        e = mid - 1
      }
      else {
        return mid
      }
    }
    val v = index(mid)
    mid = if ((greater && v >= key) || (!greater && v <= key)) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }

    if (greater) {
      if (mid < end) assert(index(mid) >= key)
      if (mid > 0) assert(index(mid - 1) <= key)
    } else {
      if (mid > 0) assert(index(mid) <= key)
      if (mid < end - 1) assert(index(mid + 1) >= key)
    }
    mid
  }
}
