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

import LDAUtils._

class LDAModel private[mllib](
  private[mllib] val gtc: BDV[Double],
  private[mllib] val ttc: Array[BSV[Double]],
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double) extends Serializable {

  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta, alpha)
  }

  @transient private lazy val numTopics = gtc.size
  @transient private lazy val numTerms = ttc.size
  @transient private lazy val numTokens = brzSum(gtc)
  @transient private lazy val betaSum = numTerms * beta
  @transient private lazy val alphaSum = numTopics * alpha
  @transient private lazy val termSum = numTokens + alphaAS * numTopics

  def globalTopicCounter = Vectors.fromBreeze(gtc)

  def topicTermCounter = ttc.map(t => Vectors.fromBreeze(t))

  def inference(
    doc: SSV,
    totalIter: Int = 10,
    burnIn: Int = 5,
    rand: Random = new Random): SSV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")

    val topicDist = BSV.zeros[Double](numTopics)
    val tokens = vector2Array(new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size))
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
    for (i <- 0 until topics.length) {
      val term = tokens(i)
      val currentTopic = topics(i)
      val newTopic = multinomialDistSampler(rand, t, w(term), d(docTopicCounter, term))
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

  private[mllib] def uniformDistSampler(
    tokens: Array[Int],
    topics: Array[Int],
    rand: Random): BSV[Double] = {
    val docTopicCounter = BSV.zeros[Double](numTopics)
    for (i <- 0 until tokens.length) {
      val topic = uniformSampler(rand, numTopics)
      topics(i) = topic
      docTopicCounter(topic) += 1D
    }
    docTopicCounter
  }

  private def multinomialDistSampler(
    rand: Random,
    t: BDV[Double],
    w: BSV[Double],
    d: BSV[Double]): Int = {
    val lastSum = t(numTopics - 1) + w.data(w.used - 1) + d.data(d.used - 1)
    val distSum = rand.nextDouble * lastSum
    val fun = index(t, w, d) _
    binarySearchInterval(fun, distSum, 0, numTopics, true)
  }

  private def index(
    t: BDV[Double],
    w: BSV[Double],
    d: BSV[Double])(topic: Int) = {
    val lastDS = binarySearchSparseVector(topic, d)
    val lastWS = binarySearchSparseVector(topic, w)
    val lastTS = t(topic)
    lastDS + lastWS + lastTS
  }

  private def d(docTopicCounter: BSV[Double], term: Int): BSV[Double] = {
    val used = docTopicCounter.used
    val index = docTopicCounter.index
    val data = docTopicCounter.data

    val d = new Array[Double](used)
    var lastSum = 0D
    var i = 0
    while (i < used) {
      val topic = index(i)
      val count = data(i)
      val lastD = count * (ttc(term)(topic) + beta) / (gtc(topic) + betaSum)
      // val lastD = count * termSum * (ttc(term)(topic) + beta) /
      // ((gtc(topic) + betaSum) * termSum)
      lastSum += lastD
      d(i) = lastSum
      i += 1
    }
    new BSV[Double](index, d, used, docTopicCounter.length)
  }

  @transient private lazy val t = {
    val t = BDV.zeros[Double](numTopics)
    var lastSum = 0D
    for (topic <- 0 until numTopics) {
      val lastT = beta * alphaSum * (gtc(topic) + alphaAS) /
        ((gtc(topic) + betaSum) * termSum)
      lastSum += lastT
      t(topic) = lastSum
    }
    t
  }

  @transient private lazy val w = {
    val w = new Array[BSV[Double]](numTerms)
    for (term <- 0 until numTerms) {
      w(term) = BSV.zeros[Double](numTopics)
      var lastSum = 0D
      ttc(term).activeIterator.foreach { case (topic, cn) =>
        val lastW = cn * alphaSum * (gtc(topic) + alphaAS) /
          ((gtc(topic) + betaSum) * termSum)
        lastSum += lastW
        w(term)(topic) = lastSum
      }
    }
    w
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
      (0 until numTerms).map(_ => BSV.zeros[Double](numTopics)).toArray, alpha, beta, alpha)
  }
}

private[mllib] object LDAUtils {

  def uniformSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
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

    // 测试代码 index(mid) 大于等于 index(mid - 1) 小于等于 index(mid + 1)
    //  if (greater) {
    //    if (mid < end) assert(ord.gteq(index(mid), key))
    //    if (mid > 0) assert(ord.lteq(index(mid - 1), key))
    //  } else {
    //    if (mid > 0) assert(ord.lteq(index(mid), key))
    //    if (mid < end - 1) assert(ord.gteq(index(mid + 1), key))
    //  }
    mid
  }

  def binarySearchArray[K](
    index: Array[K],
    key: K,
    begin: Int,
    end: Int,
    greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    binarySearchInterval(i => index(i), key, begin, end, greater)
  }

  def binarySearchSparseVector(index: Int, w: BSV[Double]) = {
    val pos = binarySearchArray(w.index, index, 0, w.used, false)
    if (pos > -1) {
      w.data(pos)
    }
    else {
      0D
    }
  }
}
