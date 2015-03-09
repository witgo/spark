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

import breeze.collection.mutable.OpenAddressHashArray
import breeze.linalg.{Vector => BV, DenseVector => BDV, HashVector => BHV,
SparseVector => BSV, sum => brzSum}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.Logging
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
  @transient var corpus: Graph[VD, ED],
  val numTopics: Int,
  val numTerms: Int,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double,
  @transient val storageLevel: StorageLevel)
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
  // scalastyle:on

  @transient private val sc = corpus.vertices.context
  @transient private val seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var totalTopicCounter: BDV[Count] = collectTotalTopicCounter(corpus)

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
    val sampledCorpus = sampleTokens(corpus, totalTopicCounter, innerIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    sampledCorpus.persist(storageLevel)

    val counterCorpus = updateCounter(sampledCorpus, numTopics)
    counterCorpus.persist(storageLevel)
    counterCorpus.vertices.count()
    counterCorpus.edges.count()
    totalTopicCounter = collectTotalTopicCounter(counterCorpus)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    sampledCorpus.edges.unpersist(false)
    sampledCorpus.vertices.unpersist(false)
    corpus = counterCorpus

    checkpoint()
    innerIter += 1
  }

  def saveModel(burnInIter: Int): LDAModel = {
    var termTopicCounter: RDD[(VertexId, VD)] = null
    for (iter <- 1 to burnInIter) {
      logInfo(s"Save TopicModel (Iteration $iter/$burnInIter)")
      var previousTermTopicCounter = termTopicCounter
      gibbsSampling()
      val newTermTopicCounter = termVertices
      termTopicCounter = Option(termTopicCounter).map(_.join(newTermTopicCounter).map {
        case (term, (a, b)) =>
          val c = new BHV(a) + new BHV(b)
          (term, c.array)
      }).getOrElse(newTermTopicCounter)

      termTopicCounter.cache().count()
      Option(previousTermTopicCounter).foreach(_.unpersist())
      previousTermTopicCounter = termTopicCounter
    }
    val model = LDAModel(numTopics, numTerms, alpha, beta)
    termTopicCounter.collect().foreach { case (term, counter) =>
      model.merge(term.toInt, new BHV(counter))
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
      // println(s"perplexity $iter: ${perplexity()}")
      logInfo(s"Start Gibbs sampling (Iteration $iter/$iterations)")
      gibbsSampling()
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
        val termTopicCounter = new BHV(counter)
        // \frac{{n}_{kw}{\alpha }_{k}}{{n}_{k}+\bar{\beta }}
        termTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * alpha /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      } else {
        val docTopicCounter = new BHV(counter)
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
      val docSize = brzSum(new BHV(docTopicCounter))
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
  private[mllib] type VD = OpenAddressHashArray[Int]
  private[mllib] type Table = Array[(Int, Int, Double)]

  def train(docs: RDD[(DocId, SSV)],
    numTopics: Int = 2048,
    totalIter: Int = 150,
    burnIn: Int = 5,
    alpha: Double = 0.1,
    beta: Double = 0.01,
    alphaAS: Double = 0.1): LDAModel = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS)
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveModel(burnIn)
  }

  def incrementalTrain(docs: RDD[(DocId, SSV)],
    computedModel: LDAModel,
    alphaAS: Double = 1,
    totalIter: Int = 150,
    burnIn: Int = 5): LDAModel = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val numTopics = computedModel.ttc.size
    val alpha = computedModel.alpha
    val beta = computedModel.beta

    val broadcastModel = docs.context.broadcast(computedModel)
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS,
      computedModel = broadcastModel)
    broadcastModel.unpersist()
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveModel(burnIn)
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
        val docTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, Table)]]()
        val wordTableCache = new AppendOnlyMap[VertexId, SoftReference[(Double, Table)]]()
        val p = tokenTopicProb(totalTopicCounter, beta, alpha,
          alphaAS, numTokens, numTerms) _
        val dPFun = docProb(totalTopicCounter, alpha, alphaAS, numTokens) _
        val wPFun = wordProb(totalTopicCounter, numTerms, beta) _

        var dD: Table = null
        var dDSum: Double = 0.0
        var wD: Table = null
        var wDSum: Double = 0.0

        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr

            if (dD == null || gen.nextDouble() < 1e-6) {
              var dv = dDense(totalTopicCounter, alpha, alphaAS, numTokens)
              dDSum = dv._1
              dD = generateAlias(dv._2, dDSum)

              dv = wDense(totalTopicCounter, numTerms, beta)
              wDSum = dv._1
              wD = generateAlias(dv._2, wDSum)
            }
            val (dSum, d) = docTopicCounter.synchronized {
              docTable(x => {
                x == null || x.get() == null || gen.nextDouble() < 1e-6
              }, docTableCache, docTopicCounter, docId)
            }
            val (wSum, w) = termTopicCounter.synchronized {
              wordTable(x => {
                x == null || x.get() == null || gen.nextDouble() < 1e-6
              }, wordTableCache, totalTopicCounter,
                termTopicCounter, termId, numTerms, beta)
            }

            for (i <- 0 until topics.length) {
              var docProposal = gen.nextDouble() < 0.5
              var maxSampling = 8
              while (maxSampling > 0) {
                maxSampling -= 1
                docProposal = !docProposal
                val currentTopic = topics(i)
                var proposalTopic = -1
                val q = if (docProposal) {
                  if (gen.nextDouble() < dDSum / (dSum - 1.0 + dDSum)) {
                    proposalTopic = sampleAlias(gen, dD)
                  }
                  else {
                    proposalTopic = docTopicCounter.synchronized {
                      sampleDoc(gen, d, docTopicCounter, currentTopic)
                    }
                  }
                  dPFun
                } else {
                  val table = if (gen.nextDouble() < wSum / (wSum + wDSum)) w else wD
                  proposalTopic = sampleAlias(gen, table)
                  wPFun
                }

                val newTopic = docTopicCounter.synchronized {
                  termTopicCounter.synchronized {
                    tokenSampling(gen, docTopicCounter, termTopicCounter, docProposal,
                      currentTopic, proposalTopic, q, p)
                  }
                }

                assert(newTopic >= 0 && newTopic < numTopics)
                if (newTopic != currentTopic) {
                  docTopicCounter.synchronized {
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                  }
                  termTopicCounter.synchronized {
                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                  }

                  totalTopicCounter(currentTopic) -= 1
                  totalTopicCounter(newTopic) += 1
                  topics(i) = newTopic
                }

              }
            }
            topics
        }
      }, TripletFields.All)
    nweGraph
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
    }, _ + _, TripletFields.EdgeOnly).mapValues { a =>
      val b = new VD(a.length)
      a.activeIterator.foreach { t =>
        b(t._1) = t._2
      }
      b
    }
    graph.joinVertices(newCounter)((_, _, nc) => nc)
  }

  private def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).
      aggregate(BDV.zeros[Count](numTopics))((a, b) => {
      a :+= new BHV(b)
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
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    corpus = corpus.partitionBy(PartitionStrategy.EdgePartition1D)
    corpus = updateCounter(corpus, numTopics).cache()
    corpus.vertices.count()
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
      doc.activeIterator.map { case (termId, counter) =>
        val ev = (0 until counter).map { i =>
          gen.nextInt(numTopics)
        }.toArray
        Edge(termId, newDocId, ev)
      }.toArray
    }
    else {
      val tokens = computedModel.vector2Array(doc)
      val topics = new Array[Int](tokens.length)
      var docTopicCounter = computedModel.uniformDistSampler(tokens, topics)
      for (t <- 0 until 15) {
        docTopicCounter = computedModel.sampleTokens(docTopicCounter,
          tokens, topics)
      }
      doc.activeIterator.map { case (term, counter) =>
        val ev = topics.zipWithIndex.filter { case (topic, offset) =>
          term == tokens(offset)
        }.map(_._1)
        Edge(term, newDocId, ev)
      }.toArray
    }
  }

  // scalastyle:off
  /**
   * 这里组合使用 Gibbs sampler 和 Metropolis Hastings sampler
   * 每次采样的复杂度为: O(1)
   * 1. 使用 Gibbs sampler 采样标准LDA公式中词相关部分:
   * 论文LightLDA: Big Topic Models on Modest Compute Clusters 公式(6):
   * ( \frac{{n}_{kd}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta }} )
   * 2. 把第一步采样得到的概率作为 Proposal q(·) 使用 Metropolis Hastings sampler 采样非对称先验公式
   * 论文 Rethinking LDA: Why Priors Matter 公式(3)
   * \frac{{n}_{kw}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta}} \frac{{n}_{kd} ^{-di}+ \bar{\alpha} \frac{{n}_{k}^{-di} + \acute{\alpha}}{\sum{n}_{k} +\bar{\acute{\alpha}}}}{\sum{n}_{kd}^{-di} +\bar{\alpha}}
   *
   * 其中
   * \bar{\beta}=\sum_{w}{\beta}_{w}
   * \bar{\alpha}=\sum_{k}{\alpha}_{k}
   * \bar{\acute{\alpha}}=\bar{\acute{\alpha}}=\sum_{k}\acute{\alpha}
   * {n}_{kd} 是文档d中主题为k的tokens数
   * {n}_{kw} 词中主题为k的tokens数
   * {n}_{k} 是语料库中主题为k的tokens数
   */
  // scalastyle:on
  def tokenSampling(
    gen: Random,
    docTopicCounter: VD,
    termTopicCounter: VD,
    docProposal: Boolean,
    currentTopic: Int,
    proposalTopic: Int,
    q: (VD, Int, Boolean) => Double,
    p: (VD, VD, Int, Boolean) => Double): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cp = p(docTopicCounter, termTopicCounter, currentTopic, true)
    val np = p(docTopicCounter, termTopicCounter, proposalTopic, false)
    val vd = if (docProposal) docTopicCounter else termTopicCounter
    val cq = q(vd, currentTopic, true)
    val nq = q(vd, proposalTopic, false)

    val pi = (np * cq) / (cp * nq)
    if (gen.nextDouble() < 1e-32) {
      println(s"Pi: ${pi}")
      println(s"($np * $cq) / ($cp * $nq)")
    }

    if (gen.nextDouble() < math.min(1.0, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  private def tokenTopicProb(
    totalTopicCounter: BDV[Count],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numTokens: Double,
    numTerms: Double)(docTopicCounter: VD,
    termTopicCounter: VD,
    topic: Int,
    isAdjustment: Boolean): Double = {
    val numTopics = docTopicCounter.length
    val adjustment = if (isAdjustment) -1 else 0
    val ratio = (totalTopicCounter(topic) + adjustment + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    // 这里移除了常数项 (docLen - 1 + alpha * numTopics)
    (termTopicCounter(topic) + adjustment + beta) *
      (docTopicCounter(topic) + adjustment + asPrior) /
      (totalTopicCounter(topic) + adjustment + (numTerms * beta))

    // 原始公式: Rethinking LDA: Why Priors Matter 公式(3)
    // val docLen = brzSum(docTopicCounter)
    // (termTopicCounter(topic) + adjustment + beta) * (docTopicCounter(topic) + adjustment + asPrior) /
    //   ((totalTopicCounter(topic) + adjustment + (numTerms * beta)) * (docLen - 1 + alpha * numTopics))
  }

  // scalastyle:on

  private def wordProb(
    totalTopicCounter: BDV[Count],
    numTerms: Double,
    beta: Double)(termTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private def docProb(
    totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Double)(docTopicCounter: VD, topic: Int, isAdjustment: Boolean): Double = {
    val adjustment = if (isAdjustment) -1.0 else 0.0
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + adjustment + asPrior
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private def wSparse(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTerms: Double,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Double](numTopics)

    var sum = 0.0
    termTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        val last = count / (totalTopicCounter(topic) + termSum)
        w(topic) = last
        sum += last
      }
    }
    (sum, w)
  }

  /**
   * \frac{{\beta}_{w}}{{n}_{k}+\bar{\beta}}
   */
  private def wDense(
    totalTopicCounter: BDV[Count],
    numTerms: Double,
    beta: Double): (Double, BV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val termSum = beta * numTerms
    var sum = 0.0
    for (topic <- 0 until numTopics) {
      val last = beta / (totalTopicCounter(topic) + termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  private def dSparse(docTopicCounter: VD): (Double, BV[Double]) = {
    val numTopics = docTopicCounter.length
    val d = BSV.zeros[Double](numTopics)
    var sum = 0.0
    docTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        val last = count
        d(topic) = last
        sum += last
      }
    }
    (sum, d)
  }


  private def dDense(
    totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Double): (Double, BV[Double]) = {
    val numTopics = totalTopicCounter.length
    val asPrior = BDV.zeros[Double](numTopics)

    var sum = 0.0
    for (topic <- 0 until numTopics) {
      val ratio = (totalTopicCounter(topic) + alphaAS) /
        (numTokens - 1 + alphaAS * numTopics)
      val last = ratio * (alpha * numTopics)
      asPrior(topic) = last
      sum += last
    }
    (sum, asPrior)
  }

  private def docTable(
    updateFunc: SoftReference[(Double, Table)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, Table)]],
    docTopicCounter: VD,
    docId: VertexId): (Double, Table) = {
    val cacheD = cacheMap(docId)
    if (!updateFunc(cacheD)) {
      cacheD.get
    } else {
      docTopicCounter.synchronized {
        val sv = dSparse(docTopicCounter)
        val d = (sv._1, generateAlias(sv._2, sv._1))
        cacheMap.update(docId, new SoftReference(d))
        d
      }
    }
  }

  private def wordTable(
    updateFunc: SoftReference[(Double, Table)] => Boolean,
    cacheMap: AppendOnlyMap[VertexId, SoftReference[(Double, Table)]],
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    termId: VertexId,
    numTerms: Double,
    beta: Double): (Double, Table) = {
    val cacheW = cacheMap(termId)
    if (!updateFunc(cacheW)) {
      cacheW.get
    } else {
      termTopicCounter.synchronized {
        val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
        val w = (sv._1, generateAlias(sv._2, sv._1))
        cacheMap.update(termId, new SoftReference(w))
        w
      }
    }
  }

  private def sampleDoc(gen: Random, table: Table, sv: VD, currentTopic: Int): Int = {
    val docTopic = sampleAlias(gen, table)
    if (docTopic == currentTopic) {
      val svCounter = sv(currentTopic)
      // 这里的处理方法不太对.
      // 如果采样到当前token的Topic这丢弃掉
      // svCounter == 1 && table.length > 1 采样到token的Topic 但包含其他token
      // svCounter > 1 && gen.nextDouble() < 1.0 / svCounter 采样的Topic 有1/svCounter 概率属于当前token
      if ((svCounter == 1 && table.length > 1) ||
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)) {
        return sampleDoc(gen, table, sv, currentTopic)
      }
    }
    docTopic
  }
}

class LDAKryoRegistrator extends KryoRegistrator {
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
