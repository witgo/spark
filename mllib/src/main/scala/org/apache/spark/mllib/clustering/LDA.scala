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
import java.util.{PriorityQueue => JPriorityQueue}
import java.util.Random

import scala.collection.mutable

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
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.util.Utils

import LDA._

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
    val sampleCorpus = sampleTokens(corpus, totalTopicCounter, innerIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
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
      println(s"perplexity $iter: ${perplexity()}")
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
    val wmap = new mutable.HashMap[VertexId, (Double, Table)]()
    val dmap = new mutable.HashMap[VertexId, (Double, Table)]()
    val broadcast = graph.edges.context.broadcast((wmap, dmap))
    val nweGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        val (wordTableMap, docTableMap) = broadcast.value
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

            var maxSampling = 6
            while (maxSampling > 0) {
              maxSampling -= 1
              for (i <- 0 until topics.length) {
                val currentTopic = topics(i)
                if (dD == null) {
                  var dv = dDense(totalTopicCounter, alpha, alphaAS, numTokens)
                  dDSum = brzSum(dv)
                  dD = generateAlias(dv)
                  dv = wDense(totalTopicCounter, numTerms, beta)
                  wDSum = brzSum(dv)
                  wD = generateAlias(dv)
                }
                val (proposalTopicFun, proposalFun) = if (gen.nextDouble() < 0.5) {
                  val d = docTableMap.synchronized {
                    docTableMap.getOrElseUpdate(docId, {
                      docTopicCounter.synchronized {
                        val sv = dSparse(docTopicCounter)
                        val sum = brzSum(sv)
                        (sum, generateAlias(sv))
                      }
                    })
                  }

                  val dSum = d._1
                  val table = if (gen.nextDouble() < dSum / (dSum + dDSum)) d._2 else dD
                  (sampleAlias(table) _, docProb(totalTopicCounter, docTopicCounter, alpha, alphaAS, numTokens) _)
                } else {
                  val w = wordTableMap.synchronized {
                    wordTableMap.getOrElseUpdate(termId, {
                      termTopicCounter.synchronized {
                        val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
                        val sum = brzSum(sv)
                        (sum, generateAlias(sv))
                      }
                    })
                  }
                  val tSum = w._1
                  val table = if (gen.nextDouble() < tSum / (tSum + wDSum)) w._2 else wD
                  (sampleAlias(table) _, wordProb(totalTopicCounter, termTopicCounter, numTerms, beta) _)
                }

                val qFun = tokenTopicProb(totalTopicCounter, docTopicCounter, termTopicCounter,
                  beta, alpha, alphaAS, numTokens, numTerms) _

                val newTopic = docTopicCounter.synchronized {
                  termTopicCounter.synchronized {
                    val proposalTopic = proposalTopicFun(gen)
                    tokenSampling(gen, currentTopic, proposalTopic, proposalFun, qFun)
                  }
                }

                if (newTopic != currentTopic) {
                  if (gen.nextDouble() < 1e-6) {
                    docTableMap.synchronized {
                      docTableMap -= docId
                    }
                  }
                  if (gen.nextDouble() < 1e-6) {
                    wordTableMap.synchronized {
                      wordTableMap -= termId
                    }
                  }
                  if (gen.nextDouble() < 1e-6) {
                    dD = null
                    wD = null
                  }

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
    wmap.clear()
    dmap.clear()
    broadcast.unpersist(false)
    nweGraph
  }

  private def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[VD](ctx => {
      val topics = ctx.attr
      val vector = BHV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector.array)
      ctx.sendToSrc(vector.array)
    }, (a, b) => {
      val c = new BHV(a) + new BHV(b)
      c.array
    }, TripletFields.EdgeOnly)
    graph.outerJoinVertices(newCounter)((_, _, n) => n.get)
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
          initializeEdges(gen, new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size),
            docId, numTopics, model)
      }
    })
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    // corpus.partitionBy(PartitionStrategy.EdgePartition1D)
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
      doc.activeIterator.map { case (term, counter) =>
        val ev = (0 until counter).map { i =>
          gen.nextInt(numTopics)
        }.toArray
        Edge(term, newDocId, ev)
      }.toArray
    }
    else {
      val tokens = computedModel.vector2Array(doc)
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
    currentTopic: Int,
    proposalTopic: Int,
    pFun: Int => Double,
    qFun: (Int, Boolean) => Double): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cq = qFun(currentTopic, true)
    val nq = qFun(proposalTopic, false)
    val cp = pFun(currentTopic)
    val np = pFun(proposalTopic)

    val pi = (nq * cp) / (cq * np)
    if (gen.nextDouble() < 1e-32) {
      println(s"Pi: ${pi}")
      println(s"($nq * $cp) / ($cq * $np)")
    }

    if (gen.nextDouble() < math.min(1.0, pi)) proposalTopic else currentTopic
  }

  // scalastyle:off
  private def tokenTopicProb(
    totalTopicCounter: BDV[Count],
    docTopicCounter: VD,
    termTopicCounter: VD,
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numTokens: Double,
    numTerms: Double)(topic: Int, isAdjustment: Boolean): Double = {
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
    termTopicCounter: VD,
    numTerms: Double,
    beta: Double)(topic: Int): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private def docProb(
    totalTopicCounter: BDV[Count],
    docTopicCounter: VD,
    alpha: Double,
    alphaAS: Double,
    numTokens: Double)(topic: Int): Double = {
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + asPrior
  }

  private def generateAlias(sv: BV[Double]): Table = {
    val used = sv.activeSize
    val svSum = brzSum(sv)
    sv :/= svSum
    generateAlias(sv.activeIterator.toArray, used)
  }

  @transient private lazy val tableOrdering = new scala.math.Ordering[(Int, Double)] {
    override def compare(x: (Int, Double), y: (Int, Double)): Int = {
      Ordering.Double.compare(x._2, y._2)
    }
  }

  private def generateAlias(
    probs: Array[(Int, Double)],
    used: Int): Table = {
    val lq = new JPriorityQueue[(Int, Double)](used, tableOrdering)
    val hq = new JPriorityQueue[(Int, Double)](used, tableOrdering.reverse)
    val pMean = 1.0 / used
    val a = new mutable.ArrayBuffer[(Int, Int, Double)]()
    for (o <- 0 until used) {
      val (i, pi) = probs(o)
      if (pi < pMean) {
        lq.add((i, pi))
      } else {
        hq.add((i, pi))
      }
    }
    while (!lq.isEmpty & !hq.isEmpty) {
      val (i, pi) = lq.remove()
      val (h, ph) = hq.remove()
      a += ((i, h, pi))
      val pd = ph - (pMean - pi)
      if (pd >= pMean) {
        hq.add((h, pd))
      } else {
        lq.add((h, pd))
      }
    }
    while (!hq.isEmpty) {
      val (h, ph) = hq.remove()
      a += ((h, h, ph))
    }

    while (!lq.isEmpty) {
      val (i, pi) = lq.remove()
      a += ((i, i, pi))
    }


    // 测试代码 随即抽样一个样本验证其概率
    val table = a.toArray
    val (di, dp) = probs(Utils.random.nextInt(probs.length))
    val ds = table.map { t =>
      if (t._1 == di) {
        if (t._2 == t._1) {
          pMean
        } else {
          t._3
        }
      } else if (t._2 == di) {
        pMean - t._3
      } else {
        0.0
      }
    }.sum
    assert((ds - dp).abs < 1e-4)
    assert(table.length == used)

    table
  }

  private def sampleAlias(table: Table)(gen: Random): Int = {
    val l = table.length
    val bin = gen.nextInt(l)
    val i = table(bin)._1
    val h = table(bin)._2
    val p = table(bin)._3
    if (l * p > gen.nextDouble()) {
      i
    } else {
      h
    }
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private def wSparse(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTerms: Double,
    beta: Double): BSV[Double] = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Double](numTopics)

    termTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        w(topic) = count / (totalTopicCounter(topic) + termSum)
      }
    }
    w
  }

  /**
   * \frac{{\beta}_{w}}{{n}_{k}+\bar{\beta}}
   */
  private def wDense(
    totalTopicCounter: BDV[Count],
    numTerms: Double,
    beta: Double): BDV[Double] = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val termSum = beta * numTerms
    for (topic <- 0 until numTopics) {
      t(topic) = beta / (totalTopicCounter(topic) + termSum)
    }
    t
  }

  private def dDense(
    totalTopicCounter: BDV[Count],
    alpha: Double,
    alphaAS: Double,
    numTokens: Double): BDV[Double] = {
    val numTopics = totalTopicCounter.length
    val asPrior = BDV.zeros[Double](numTopics)

    for (topic <- 0 until numTopics) {
      val ratio = (totalTopicCounter(topic) + alphaAS) /
        (numTokens - 1 + alphaAS * numTopics)
      asPrior(topic) = ratio * (alpha * numTopics)
    }
    asPrior
  }

  private def dSparse(docTopicCounter: VD): BSV[Double] = {
    val numTopics = docTopicCounter.length
    val d = BSV.zeros[Double](numTopics)
    docTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        d(topic) = count
      }
    }
    d
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
