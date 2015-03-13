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

import scala.collection.mutable
import scala.reflect.ClassTag

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.{HashPartitioner, Partitioner, Logging}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext._
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.util.random.XORShiftRandom

import LDA._
import LDAUtils._

class LDA private[mllib](
  @transient private var corpus: Graph[VD, ED],
  private val numTopics: Int,
  private val numTerms: Int,
  private var alpha: Double,
  private var beta: Double,
  private var alphaAS: Double,
  private var storageLevel: StorageLevel)
  extends Serializable with Logging {

  def this(docs: RDD[(DocId, SSV)],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    computedModel: Broadcast[LDAModel] = null) {
    this(initializeCorpus(docs, numTopics, storageLevel, computedModel),
      numTopics, docs.first()._2.size, alpha, beta, alphaAS, storageLevel)
  }

  val numDocs = docVertices.count()
  val numTokens = corpus.edges.map(e => e.attr.size.toDouble).sum().toLong

  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }

  def setBeta(beta: Double): this.type = {
    this.beta = beta
    this
  }

  def setAlphaAS(alphaAS: Double): this.type = {
    this.alphaAS = alphaAS
    this
  }

  def setStorageLevel(newStorageLevel: StorageLevel): this.type = {
    this.storageLevel = newStorageLevel
    this
  }

  def setSeed(newSeed: Int): this.type = {
    this.seed = newSeed
    this
  }

  @transient private val sc = corpus.vertices.context
  @transient private var seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var totalTopicCounter: BDV[Count] = collectTotalTopicCounter(corpus)

  def getCorpus: Graph[VD, ED] = corpus

  private def termVertices = corpus.vertices.filter(t => t._1 >= 0)

  private def docVertices = corpus.vertices.filter(t => t._1 < 0)

  private def checkpoint(): Unit = {
    if (innerIter % 10 == 0 && sc.getCheckpointDir.isDefined) {
      val edges = corpus.edges.map(t => t)
      edges.checkpoint()
      val newCorpus: Graph[VD, ED] = Graph.fromEdges(edges, null,
        storageLevel, storageLevel)
      corpus = updateCounter(newCorpus, numTopics).persist(storageLevel)
    }
  }

  private def collectTotalTopicCounter(graph: Graph[VD, ED]): BDV[Count] = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == numTokens)
    globalTopicCounter
  }

  private def gibbsSampling(): Unit = {
    val sampleCorpus = sampleTokens(corpus, totalTopicCounter,
      innerIter + seed, numTokens, numTerms, numTopics, alpha, beta, alphaAS)
    sampleCorpus.persist(storageLevel)

    val counterCorpus = updateCounter(sampleCorpus, numTopics)
    counterCorpus.persist(storageLevel)
    // counterCorpus.vertices.count()
    counterCorpus.edges.count()
    totalTopicCounter = collectTotalTopicCounter(counterCorpus)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    sampleCorpus.edges.unpersist(false)
    sampleCorpus.vertices.unpersist(false)
    corpus = counterCorpus

    checkpoint()
    innerIter += 1
  }

  def saveModel(burnInIter: Int): LDAModel = {
    var termTopicCounter: RDD[(Int, BSV[Double])] = null
    for (iter <- 1 to burnInIter) {
      logInfo(s"Save TopicModel (Iteration $iter/$burnInIter)")
      var previousTermTopicCounter = termTopicCounter
      gibbsSampling()
      val newTermTopicCounter = updateModel(termVertices)
      termTopicCounter = Option(termTopicCounter).map(_.join(newTermTopicCounter).map {
        case (term, (a, b)) =>
          val c = a + b
          c.compact()
          (term, c)
      }).getOrElse(newTermTopicCounter)

      termTopicCounter.cache().count()
      Option(previousTermTopicCounter).foreach(_.unpersist())
      previousTermTopicCounter = termTopicCounter
    }
    val model = LDAModel(numTopics, numTerms, alpha, beta)
    termTopicCounter.collect().foreach { case (term, counter) =>
      model.merge(term, counter)
    }
    model.gtc :/= burnInIter.toDouble
    model.ttc.foreach { ttc =>
      ttc :/= burnInIter.toDouble
      ttc.compact()
    }
    model
  }

  def runGibbsSampling(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      // println(s"perplexity $iter:                 ${perplexity}")
      logInfo(s"Start Gibbs sampling (Iteration $iter/$iterations)")
      gibbsSampling()
    }
  }

  def mergeDuplicateTopic(threshold: Double = 0.95D): Map[Int, Int] = {
    val rows = termVertices.map(t => t._2).map { bsv =>
      val length = bsv.length
      val used = bsv.used
      val index = bsv.index.slice(0, used)
      val data = bsv.data.slice(0, used).map(_.toDouble)
      new SSV(length, index, data).asInstanceOf[SV]
    }
    val simMatrix = new RowMatrix(rows).columnSimilarities()
    val minMap = simMatrix.entries.filter { case MatrixEntry(row, column, sim) =>
      sim > threshold && row != column
    }.map { case MatrixEntry(row, column, sim) =>
      (column.toInt, row.toInt)
    }.groupByKey().map { case (topic, simTopics) =>
      (topic, simTopics.min)
    }.collect().toMap
    if (minMap.size > 0) {
      corpus = corpus.mapEdges(edges => {
        edges.attr.map { topic =>
          minMap.get(topic).getOrElse(topic)
        }
      })
      corpus = updateCounter(corpus, numTopics)
    }
    minMap
  }

  def perplexity(): Double = {
    val totalTopicCounter = this.totalTopicCounter
    val numTopics = this.numTopics
    val numTerms = this.numTerms
    val alpha = this.alpha
    val beta = this.beta
    val totalSize = brzSum(totalTopicCounter)
    var totalProb = 0D

    totalTopicCounter.activeIterator.foreach { case (topic, cn) =>
      totalProb += alpha * beta / (cn + numTerms * beta)
    }

    val termProb = corpus.mapVertices { (vid, counter) =>
      val probDist = BSV.zeros[Double](numTopics)
      if (vid >= 0) {
        val termTopicCounter = counter
        termTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * alpha /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      } else {
        val docTopicCounter = counter
        docTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * beta /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      }
      probDist.compact()
      (counter, probDist)
    }.mapTriplets { triplet =>
      val (termTopicCounter, termProb) = triplet.srcAttr
      val (docTopicCounter, docProb) = triplet.dstAttr
      val docSize = brzSum(docTopicCounter)
      val docTermSize = triplet.attr.length

      var prob = 0D

      docTopicCounter.activeIterator.foreach { case (topic, cn) =>
        prob += cn * termTopicCounter(topic) /
          (totalTopicCounter(topic) + numTerms * beta)
      }
      prob += brzSum(docProb) + brzSum(termProb) + totalProb
      prob += prob / (docSize + numTopics * alpha)

      docTermSize * Math.log(prob)
    }.edges.map(t => t.attr).sum()

    math.exp(-1 * termProb / totalSize)
  }
}

object LDA {

  import LDAUtils._

  private[mllib] type DocId = VertexId
  private[mllib] type WordId = VertexId
  private[mllib] type Count = Int
  private[mllib] type ED = Array[Count]
  private[mllib] type VD = BSV[Count]

  def train(docs: RDD[(DocId, SSV)],
    numTopics: Int = 2048,
    totalIter: Int = 150,
    alpha: Double = 0.1,
    beta: Double = 0.01,
    alphaAS: Double = 0.1): LDAModel = {
    require(totalIter > 0, "totalIter is less than 0")
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS)
    topicModeling.runGibbsSampling(totalIter - 1)
    topicModeling.saveModel(1)
  }

  def incrementalTrain(docs: RDD[(DocId, SSV)],
    computedModel: LDAModel,
    alphaAS: Double = 1,
    totalIter: Int = 150): LDAModel = {
    require(totalIter > 0, "totalIter is less than 0")
    val numTopics = computedModel.ttc.size
    val alpha = computedModel.alpha
    val beta = computedModel.beta

    val broadcastModel = docs.context.broadcast(computedModel)
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS,
      computedModel = broadcastModel)
    broadcastModel.unpersist()
    topicModeling.runGibbsSampling(totalIter - 1)
    topicModeling.saveModel(1)
  }

  private[mllib] def sampleTokens(
    graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    innerIter: Long,
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        val d = BDV.zeros[Double](numTopics)
        val wCache = new AppendOnlyMap[VertexId, SoftReference[BSV[Double]]]()
        var tCache: BDV[Double] = null
        iter.map {
          triplet =>
            val termId = triplet.srcId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr
            termTopicCounter.synchronized {
              docTopicCounter.synchronized {
                val w = this.w(x => x == null || x.get() == null || gen.nextDouble() < 1e-4,
                  wCache, totalTopicCounter, termTopicCounter, termId,
                  numTokens, numTerms, alpha, beta, alphaAS)

                if (tCache == null || gen.nextDouble() < 1e-7) {
                  tCache = LDA.t(totalTopicCounter, numTokens, numTerms,
                    numTopics, alpha, beta, alphaAS)
                }
                val t = tCache
                var i = 0
                while (i < topics.length) {
                  val currentTopic = topics(i)
                  this.d(totalTopicCounter, termTopicCounter, docTopicCounter, d,
                    currentTopic, numTokens, numTerms, numTopics, beta, alphaAS)
                  val newTopic = multinomialDistSampler(gen, docTopicCounter, d, w, t)
                  if (currentTopic != newTopic) {
                    topics(i) = newTopic
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                    if (docTopicCounter(currentTopic) == 0) docTopicCounter.compact()

                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                    if (termTopicCounter(currentTopic) == 0) termTopicCounter.compact()

                    totalTopicCounter(currentTopic) -= 1
                    totalTopicCounter(newTopic) += 1


                  }
                  i += 1
                }
              }
            }
            topics
        }
      }, TripletFields.All)
  }

  private def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[BSV[Count]](ctx => {
      val topics = ctx.attr
      val vector = BSV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector)
      ctx.sendToSrc(vector)
    }, (a, b) => {
      val c = a + b
      c.compact()
      c
    }, TripletFields.EdgeOnly)
    graph.outerJoinVertices(newCounter)((_, _, n) => n.get)
  }

  private def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).
      aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  }

  private def updateModel(termVertices: VertexRDD[VD]): RDD[(Int, BSV[Double])] = {
    termVertices.map(vertex => {
      val termTopicCounter = vertex._2
      val index = termTopicCounter.index.slice(0, termTopicCounter.used)
      val data = termTopicCounter.data.slice(0, termTopicCounter.used).map(_.toDouble)
      val used = termTopicCounter.used
      val length = termTopicCounter.length
      (vertex._1.toInt, new BSV[Double](index, data, used, length))
    })
  }

  private def initializeCorpus(
    docs: RDD[(LDA.DocId, SSV)],
    numTopics: Int,
    storageLevel: StorageLevel,
    computedModel: Broadcast[LDAModel] = null): Graph[VD, ED] = {
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      var model: LDAModel = null
      if (computedModel != null) model = computedModel.value
      iter.flatMap {
        case (docId, doc) =>
          val bsv = new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size)
          initializeEdges(gen, bsv, docId, numTopics, model)
      }
    })
    edges.persist(storageLevel)
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    // degree-based hashing
    val degrees = corpus.outerJoinVertices(corpus.degrees) { (vid, data, deg) => deg.getOrElse(0)}
    val numPartitions = edges.partitions.size
    val partitionStrategy = new DBHPartitioner(numPartitions)
    val newEdges = degrees.triplets.map { e =>
      (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    corpus = Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
    // end degree-based hashing
    // corpus = corpus.partitionBy(PartitionStrategy.EdgePartition2D)
    corpus = updateCounter(corpus, numTopics).cache()
    corpus.vertices.count()
    corpus.edges.count()
    edges.unpersist()
    corpus
  }

  private def initializeEdges(
    gen: Random,
    doc: BSV[Int],
    docId: DocId,
    numTopics: Int,
    computedModel: LDAModel = null): Array[Edge[ED]] = {
    assert(docId >= 0)
    val newDocId: DocId = -(docId + 1L)
    if (computedModel == null) {
      doc.activeIterator.map { case (term, counter) =>
        val ev = (0 until counter).map { i =>
          uniformSampler(gen, numTopics)
        }.toArray
        Edge(term, newDocId, ev)
      }.toArray
    }
    else {
      val tokens = computedModel.vec2Array(doc)
      val topics = new Array[Int](tokens.length)
      var docTopicCounter = computedModel.uniformDistSampler(tokens, topics, gen)
      for (t <- 0 until 15) {
        docTopicCounter = computedModel.sampleTokens(docTopicCounter,
          tokens, topics, gen)
      }
      doc.activeIterator.map { case (term, counter) =>
        val ev = topics.zipWithIndex.filter { case (topic, offset) =>
          term == tokens(offset)
        }.map(_._1)
        Edge(term, newDocId, ev)
      }.toArray
    }
  }

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  private def multinomialDistSampler(
    gen: Random,
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    w: BSV[Double],
    t: BDV[Double]): Int = {
    val numTopics = d.length
    val lastSum = t(numTopics - 1) + w.data(w.used - 1) + d(numTopics - 1)
    val distSum = gen.nextDouble() * lastSum
    val fun = index(docTopicCounter, d, w, t) _
    val topic = binarySearchInterval(fun, distSum, 0, numTopics, true)
    math.min(topic, numTopics - 1)
  }

  private def index(
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    w: BSV[Double],
    t: BDV[Double])(i: Int) = {
    val lastDS = binarySearchDenseVector(i, docTopicCounter, d)
    val lastWS = binarySearchSparseVector(i, w)
    val lastTS = t(i)
    lastDS + lastWS + lastTS
  }

  private def d(
    totalTopicCounter: BDV[Count],
    termTopicCounter: BSV[Count],
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    currentTopic: Int,
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    beta: Double,
    alphaAS: Double): Unit = {
    val used = docTopicCounter.used
    val index = docTopicCounter.index
    val data = docTopicCounter.data

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var i = 0
    var lastSum = 0D

    while (i < used) {
      val topic = index(i)
      val count: Double = data(i)
      val adjustment = if (currentTopic == topic) -1D else 0
      // val lastD = count * termSum * (termTopicCounter(topic) + beta) /
      //   ((totalTopicCounter(topic) + betaSum) * termSum)

      val lastD = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      lastSum += lastD
      d(topic) = lastSum
      i += 1
    }
    d(numTopics - 1) = lastSum
  }

  private def w(
    updateFunc: SoftReference[BSV[Double]] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[BSV[Double]]],
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTokens: Long,
    numTerms: Double,
    alpha: Double,
    beta: Double,
    alphaAS: Double): BSV[Double] = {
    val cacheW = cacheMap(termId)
    if (!updateFunc(cacheW)) {
      cacheW.get
    } else {
      val numTopics = totalTopicCounter.length
      val alphaSum = alpha * numTopics
      val termSum = numTokens - 1D + alphaAS * numTopics
      val betaSum = numTerms * beta
      val length = termTopicCounter.length
      val used = termTopicCounter.used
      val index = termTopicCounter.index.slice(0, used)
      val data = termTopicCounter.data
      val w = new Array[Double](used)

      var lastSum = 0D
      var i = 0

      while (i < used) {
        val topic = index(i)
        val count = data(i)
        val lastW = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
          ((totalTopicCounter(topic) + betaSum) * termSum)
        lastSum += lastW
        w(i) = lastSum
        i += 1
      }
      val sv = new BSV[Double](index, w, used, length)
      cacheMap.update(termId, new SoftReference(sv))
      sv
    }
  }

  private def t(
    totalTopicCounter: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double): BDV[Double] = {
    val t = BDV.zeros[Double](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta

    var lastSum = 0D
    for (topic <- 0 until numTopics) {
      val lastT = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      lastSum += lastT
      t(topic) = lastSum
    }
    t
  }
}

object LDAUtils {

  private[mllib] def uniformSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }

  private[mllib] def binarySearchInterval[K](
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

    if (greater) {
      if (mid < end) assert(ord.gteq(index(mid), key))
      if (mid > 0) assert(ord.lteq(index(mid - 1), key))
    } else {
      if (mid > 0) assert(ord.lteq(index(mid), key))
      if (mid < end - 1) assert(ord.gteq(index(mid + 1), key))
    }
    mid
  }

  private[mllib] def binarySearchArray[K](
    index: Array[K],
    key: K,
    begin: Int,
    end: Int,
    greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    binarySearchInterval(i => index(i), key, begin, end, greater)
  }

  private[mllib] def binarySearchSparseVector(topic: Int, sv: BSV[Double]) = {
    val index = sv.index
    val used = sv.used
    val data = sv.data
    val pos = binarySearchArray(index, topic, 0, used, false)
    if (pos > -1) data(pos) else 0D
  }

  private[mllib] def binarySearchDenseVector[V](
    topic: Int,
    sv: BSV[V],
    dv: BDV[Double]): Double = {
    val index = sv.index
    val used = sv.used
    val pos = binarySearchArray(index, topic, 0, used, false)
    if (pos > -1) dv(index(pos)) else 0D
  }

}

/**
 * Degree-Based Hashing, the paper:
 * http://nips.cc/Conferences/2014/Program/event.php?ID=4569
 * @param partitions
 */
private class DBHPartitioner(partitions: Int) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  def numPartitions = partitions

  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
    val idx = math.min(edge.srcAttr, edge.dstAttr)
    getPartition(idx)
  }

  def getPartition(idx: Int): PartitionID = {
    (math.abs(idx * mixingPrime) % partitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case h: DBHPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

private[mllib] class LDAKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    val gkr = new GraphKryoRegistrator
    gkr.registerClasses(kryo)

    kryo.register(classOf[BSV[LDA.Count]])
    kryo.register(classOf[BSV[Double]])

    kryo.register(classOf[BDV[LDA.Count]])
    kryo.register(classOf[BDV[Double]])

    kryo.register(classOf[SV])
    kryo.register(classOf[SSV])
    kryo.register(classOf[SDV])

    kryo.register(classOf[LDA.ED])
    kryo.register(classOf[LDA.VD])

    kryo.register(classOf[Random])
    kryo.register(classOf[LDA])
    kryo.register(classOf[LDAModel])
  }
}
