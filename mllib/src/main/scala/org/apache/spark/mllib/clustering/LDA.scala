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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.{HashPartitioner, Logging, Partitioner}
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

  // scalastyle:off
  /**
   * 语料库文档数
   */
  val numDocs = docVertices.count()

  /**
   * 语料库总的词数(包含重复)
   */
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

  def getCorpus = corpus

  // scalastyle:on

  @transient private var seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var totalTopicCounter: BDV[Count] = collectTotalTopicCounter(corpus)

  private def termVertices = corpus.vertices.filter(t => t._1 >= 0)

  private def docVertices = corpus.vertices.filter(t => t._1 < 0)

  private def checkpoint(corpus: Graph[VD, ED]): Unit = {
    if (innerIter % 10 == 0 && corpus.edges.sparkContext.getCheckpointDir.isDefined) {
      corpus.checkpoint()
    }
  }

  private def collectTotalTopicCounter(graph: Graph[VD, ED]): BDV[Count] = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == numTokens)
    globalTopicCounter
  }

  private def gibbsSampling(): Unit = {
    val sampledCorpus = sampleTokens(corpus, totalTopicCounter, innerIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    sampledCorpus.persist(storageLevel)

    val counterCorpus = updateCounter(sampledCorpus, numTopics)
    checkpoint(counterCorpus)
    counterCorpus.persist(storageLevel)
    // counterCorpus.vertices.count()
    counterCorpus.edges.count()
    totalTopicCounter = collectTotalTopicCounter(counterCorpus)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    sampledCorpus.edges.unpersist(false)
    sampledCorpus.vertices.unpersist(false)
    corpus = counterCorpus
    innerIter += 1
  }

  def saveModel(iter: Int = 1): LDAModel = {
    var termTopicCounter: RDD[(VertexId, VD)] = null
    for (iter <- 1 to iter) {
      logInfo(s"Save TopicModel (Iteration $iter/$iter)")
      var previousTermTopicCounter = termTopicCounter
      gibbsSampling()
      val newTermTopicCounter = termVertices
      termTopicCounter = Option(termTopicCounter).map(_.join(newTermTopicCounter).map {
        case (term, (a, b)) =>
          (term, a :+ b)
      }).getOrElse(newTermTopicCounter)

      termTopicCounter.persist(storageLevel).count()
      Option(previousTermTopicCounter).foreach(_.unpersist())
      previousTermTopicCounter = termTopicCounter
    }
    val model = LDAModel(numTopics, numTerms, alpha, beta)
    termTopicCounter.collect().foreach { case (term, counter) =>
      model.merge(term.toInt, counter)
    }
    model.gtc :/= iter.toDouble
    model.ttc.foreach { ttc =>
      ttc :/= iter.toDouble
      ttc.compact()
    }
    model
  }

  def runGibbsSampling(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      // logInfo(s"Gibbs samplin perplexity $iter:                 ${perplexity}")
      // logInfo(s"Gibbs sampling (Iteration $iter/$iterations)")
      // val startedAt = System.nanoTime()
      gibbsSampling()
      // val endAt = System.nanoTime()
      // val useTime = (endAt - startedAt) / 1e9
      // logInfo(s"Gibbs sampling use time  $iter:              $useTime")
    }
  }

  def mergeDuplicateTopic(threshold: Double = 0.95D): Map[Int, Int] = {
    val rows = termVertices.map(t => t._2).map { bsv =>
      val length = bsv.length
      val used = bsv.activeSize
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


  // scalastyle:off
  /**
   * 词在所有主题分布和该词所在文本的主题分布乘积: p(w)=\sum_{k}{p(k|d)*p(w|k)}=
   * \sum_{k}{\frac{{n}_{kw}+{\beta }_{w}} {{n}_{k}+\bar{\beta }} \frac{{n}_{kd}+{\alpha }_{k}} {\sum{{n}_{k}}+\bar{\alpha }}}=
   * \sum_{k} \frac{{\alpha }_{k}{\beta }_{w}  + {n}_{kw}{\alpha }_{k} + {n}_{kd}{\beta }_{w} + {n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta }} \frac{1}{\sum{{n}_{k}}+\bar{\alpha }}}
   * \exp^{-(\sum{\log(p(w))})/N}
   * N为语料库包含的token数
   */
  // scalastyle:on
  def perplexity(): Double = {
    val totalTopicCounter = this.totalTopicCounter
    val numTopics = this.numTopics
    val numTerms = this.numTerms
    val alpha = this.alpha
    val beta = this.beta
    val totalSize = brzSum(totalTopicCounter)
    var totalProb = 0D

    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    totalTopicCounter.activeIterator.foreach { case (topic, cn) =>
      totalProb += alpha * beta / (cn + numTerms * beta)
    }

    val termProb = corpus.mapVertices { (vid, counter) =>
      val probDist = BSV.zeros[Double](numTopics)
      if (vid >= 0) {
        val termTopicCounter = counter
        // \frac{{n}_{kw}{\alpha }_{k}}{{n}_{k}+\bar{\beta }}
        termTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * alpha /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      } else {
        val docTopicCounter = counter
        // \frac{{n}_{kd}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
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

      // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
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

  private[mllib] type DocId = VertexId
  private[mllib] type WordId = VertexId
  private[mllib] type Count = Int
  private[mllib] type ED = Array[Count]
  private[mllib] type VD = BSV[Count]

  def train(docs: RDD[(DocId, SSV)],
    numTopics: Int = 2048,
    totalIter: Int = 150,
    alpha: Double = 0.01,
    beta: Double = 0.01,
    alphaAS: Double = 0.1): LDAModel = {
    require(totalIter > 0, "totalIter is less than 0")
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS)
    topicModeling.runGibbsSampling(totalIter - 1)
    topicModeling.saveModel(1)
  }

  /**
   * topicID termID+1:counter termID+1:counter ..
   */
  def trainAndSaveModel(
    docs: RDD[(DocId, SSV)],
    dir: String,
    numTopics: Int = 2048,
    totalIter: Int = 150,
    alpha: Double = 0.01,
    beta: Double = 0.01,
    alphaAS: Double = 0.1): Unit = {
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.util.MLUtils
    import org.apache.spark.mllib.linalg.Vectors
    val lda = new LDA(docs, numTopics, alpha, beta, alphaAS)
    val numTerms = lda.numTerms
    lda.runGibbsSampling(totalIter)
    val rdd = lda.termVertices.flatMap { case (termId, counter) =>
      counter.activeIterator.map { case (topic, cn) =>
        val sv = BSV.zeros[Double](numTerms)
        sv(termId.toInt) = cn.toDouble
        (topic, sv)
      }
    }.reduceByKey { (a, b) => a + b }.map { case (topic, sv) =>
      LabeledPoint(topic.toDouble, Vectors.fromBreeze(sv))
    }
    MLUtils.saveAsLibSVMFile(rdd, dir)
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
    numTokens: Double,
    numTopics: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val nweGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        val wordTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, Table)]]()
        val dv = tDense(totalTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
        val dData = new Array[Double](numTopics.toInt)
        val t = generateAlias(dv._2, dv._1)
        val tSum = dv._1
        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr
            for (i <- 0 until topics.length) {
              val currentTopic = topics(i)
              docTopicCounter.synchronized {
                termTopicCounter.synchronized {
                  dSparse(totalTopicCounter, termTopicCounter, docTopicCounter, dData,
                    currentTopic, numTokens, numTerms, alpha, alphaAS, beta)
                  val (wSum, w) = wordTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-4,
                    wordTableCache, totalTopicCounter, termTopicCounter,
                    termId, numTokens, numTerms, alpha, alphaAS, beta)
                  val newTopic = tokenSampling(gen, t, tSum, w, termTopicCounter, wSum,
                    docTopicCounter, dData, currentTopic)

                  if (newTopic != currentTopic) {
                    topics(i) = newTopic
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                    // if (docTopicCounter(currentTopic) == 0) docTopicCounter.compact()

                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                    // if (termTopicCounter(currentTopic) == 0) termTopicCounter.compact()

                    totalTopicCounter(currentTopic) -= 1
                    totalTopicCounter(newTopic) += 1
                  }
                }
              }
            }

            topics
        }
      }, TripletFields.All)
    nweGraph
  }

  private def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[VD](ctx => {
      val topics = ctx.attr
      val vector = BSV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector)
      ctx.sendToSrc(vector)
    }, _ + _, TripletFields.EdgeOnly).mapValues(v => {
      val used = v.used
      if (v.index.length == used) {
        v
      } else {
        val index = new Array[Int](used)
        val data = new Array[Count](used)
        Array.copy(v.index, 0, index, 0, used)
        Array.copy(v.data, 0, data, 0, used)
        new BSV[Count](index, data, numTopics)
      }
    })
    // GraphImpl.fromExistingRDDs(newCounter, graph.edges)
    GraphImpl(newCounter, graph.edges)
  }

  private def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).
      aggregate(BDV.zeros[Count](numTopics))((a, b) => {
      a :+= b
    }, _ :+= _)
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
    val degrees = corpus.outerJoinVertices(corpus.degrees) { (vid, data, deg) => deg.getOrElse(0) }
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
    val edges = if (computedModel == null) {
      doc.activeIterator.filter(_._2 > 0).map { case (termId, counter) =>
        val topics = new Array[Int](counter)
        for (i <- 0 until counter) {
          topics(i) = gen.nextInt(numTopics)
        }
        Edge(termId, newDocId, topics)
      }.toArray
    }
    else {
      computedModel.setSeed(gen.nextInt())
      val tokens = computedModel.vector2Array(doc)
      val topics = new Array[Int](tokens.length)
      var docTopicCounter = computedModel.uniformDistSampler(tokens, topics)
      for (t <- 0 until 15) {
        docTopicCounter = computedModel.sampleTokens(docTopicCounter,
          tokens, topics)
      }
      doc.activeIterator.filter(_._2 > 0).map { case (term, counter) =>
        val ev = topics.zipWithIndex.filter { case (topic, offset) =>
          term == tokens(offset)
        }.map(_._1)
        Edge(term, newDocId, ev)
      }.toArray
    }
    assert(edges.length > 0)
    edges
  }

  // scalastyle:off
  /**
   * 这里组合使用 Gibbs sampler 和 Metropolis Hastings sampler
   * 每次采样的复杂度为: O(1)
   * 使用 Gibbs sampler 采样论文 Rethinking LDA: Why Priors Matter 公式(3)
   * \frac{{n}_{kw}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta}} \frac{{n}_{kd} ^{-di}+ \bar{\alpha} \frac{{n}_{k}^{-di} + \acute{\alpha}}{\sum{n}_{k} +\bar{\acute{\alpha}}}}{\sum{n}_{kd}^{-di} +\bar{\alpha}}
   * = t + w + d
   * t 全局相关部分
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta}) ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * w 词相关部分
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * d 文档和词的乘积
   * d =  \frac{{n}_{kd}^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   * 其中
   * \bar{\beta}=\sum_{w}{\beta}_{w}
   * \bar{\alpha}=\sum_{k}{\alpha}_{k}
   * \bar{\acute{\alpha}}=\bar{\acute{\alpha}}=\sum_{k}\acute{\alpha}
   * {n}_{kd} 文档d中主题为k的tokens数
   * {n}_{kw} 词中主题为k的tokens数
   * {n}_{k} 语料库中主题为k的tokens数
   * -di 减去当前token的主题
   */
  // scalastyle:on
  private def tokenSampling(
    gen: Random,
    t: Table,
    tSum: Double,
    w: Table,
    termTopicCounter: VD,
    wSum: Double,
    docTopicCounter: VD,
    dData: Array[Double],
    currentTopic: Int): Int = {
    val index = docTopicCounter.index
    val used = docTopicCounter.used
    val dSum = dData(docTopicCounter.used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextDouble() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextDouble() * dSum
      val pos = binarySearchInterval(dData, dGenSum, 0, used, true)
      index(pos)
    } else if (genSum < (dSum + wSum)) {
      sampleSV(gen, w, termTopicCounter, currentTopic)
    } else {
      sampleAlias(gen, t)
    }
  }


  /**
   * 分解后的公式为
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta}) ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def tDense(
    totalTopicCounter: BDV[Count],
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BDV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0
    for (topic <- 0 until numTopics) {
      val last = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  /**
   * 分解后的公式为
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta}) ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def wSparse(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, BSV[Double]) = {
    val numTopics = totalTopicCounter.length
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    val w = BSV.zeros[Double](numTopics)
    var sum = 0.0
    termTopicCounter.activeIterator.filter(_._2 > 0).foreach { t =>
      val topic = t._1
      val count = t._2
      val last = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      w(topic) = last
      sum += last
    }
    (sum, w)
  }

  /**
   * 分解后的公式为
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  private def dSparse(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    docTopicCounter: VD,
    d: Array[Double],
    currentTopic: Int,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): Unit = {
    val index = docTopicCounter.index
    val data = docTopicCounter.data
    val used = docTopicCounter.used

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0
    for (i <- 0 until used) {
      val topic = index(i)
      val count = data(i)
      val adjustment = if (currentTopic == topic) -1D else 0
      val last = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      // val lastD = (count + adjustment) * termSum * (termTopicCounter(topic) + adjustment + beta) /
      //  ((totalTopicCounter(topic) + adjustment + betaSum) * termSum)

      sum += last
      d(i) = sum
    }
  }

  private def wordTable(
    updateFunc: SoftReference[(Double, Table)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, Table)]],
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTokens: Double,
    numTerms: Double,
    alpha: Double,
    alphaAS: Double,
    beta: Double): (Double, Table) = {
    val cacheW = cacheMap(termId)
    if (!updateFunc(cacheW)) {
      cacheW.get
    } else {
      val sv = wSparse(totalTopicCounter, termTopicCounter,
        numTokens, numTerms, alpha, alphaAS, beta)
      val w = (sv._1, generateAlias(sv._2, sv._1))
      cacheMap.update(termId, new SoftReference(w))
      w
    }
  }

  private def sampleSV(gen: Random, table: Table, sv: VD, currentTopic: Int): Int = {
    val docTopic = sampleAlias(gen, table)
    if (docTopic == currentTopic) {
      val svCounter = sv(currentTopic)
      // 这里的处理方法不太对.
      // 如果采样到当前token的Topic这丢弃掉
      // svCounter == 1 && table.length > 1 采样到token的Topic 但包含其他token
      // svCounter > 1 && gen.nextDouble() < 1.0 / svCounter 采样的Topic 有1/svCounter 概率属于当前token
      if ((svCounter == 1 && table._1.length > 1) ||
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)) {
        return sampleSV(gen, table, sv, currentTopic)
      }
    }
    docTopic
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
    val srcDeg = edge.srcAttr
    val dstDeg = edge.dstAttr
    val srcId = edge.srcId
    val dstId = edge.dstId
    val minId = if (srcDeg < dstDeg) srcId else dstId
    getPartition(minId)
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
