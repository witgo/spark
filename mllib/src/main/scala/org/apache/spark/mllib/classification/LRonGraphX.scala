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

package org.apache.spark.mllib.classification

import scala.math._

import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.mllib.classification.LRonGraphX._
import org.apache.spark.mllib.linalg.{DenseVector => SDV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.{HashPartitioner, Logging, Partitioner}

class LRonGraphX(
  @transient var dataSet: Graph[VD, ED],
  val numFeatures: Int,
  val stepSize: Double,
  val regParam: Double,
  val epsilon: Double,
  @transient var storageLevel: StorageLevel) extends Serializable with Logging {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    numFeatures: Int,
    stepSize: Double = 1e-4,
    regParam: Double = 0.0,
    epsilon: Double = 0.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, storageLevel),
      numFeatures, stepSize, regParam, epsilon, storageLevel)
  }

  if (dataSet.vertices.getStorageLevel == StorageLevel.NONE) {
    dataSet.persist(storageLevel)
  }

  @transient private var innerIter = 1
  lazy val numSamples: Long = samples.count()

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 < 0)
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 >= 0)
  }

  // Modified Iterative Scaling, the paper:
  // A comparison of numerical optimizers for logistic regression
  // http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf
  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      val previousDataSet = dataSet
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val q = forward()
      logInfo(s"train (Iteration $iter/$iterations) Log likelihood : ${logLikelihood(q)}")
      val delta = backward(q)
      dataSet = updateWeight(delta, iter)
      dataSet = checkpoint(dataSet)
      if (dataSet.vertices.getStorageLevel == StorageLevel.NONE) {
        dataSet.persist(storageLevel)
      }
      dataSet.vertices.count()
      dataSet.edges.count()
      previousDataSet.unpersist()
      logInfo(s"End train (Iteration $iter/$iterations)")
      innerIter += 1
    }
  }

  def saveModel(): LogisticRegressionModel = {
    val featureData = new Array[Double](numFeatures)
    features.toLocalIterator.foreach { case (index, value) =>
      featureData(index.toInt) = value
    }
    new LogisticRegressionModel(new SDV(featureData), 0.0)
  }

  def logLikelihood(q: VertexRDD[VD]): Double = {
    samples.join(q).map { case (_, (y, q)) =>
      val score = 1.0 - q
      val label = if (y > 0.0) 1.0 else 0.0
      if (label == 1.0) {
        math.log(score)
      } else {
        math.log(1 - score)
      }
    }.reduce(_ + _) / numSamples
  }

  def forward(): VertexRDD[VD] = {
    dataSet.aggregateMessages[Double](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val w = ctx.srcAttr
      val y = ctx.dstAttr
      val z = y * w * x
      assert(!z.isNaN)
      ctx.sendToDst(z)
    }, _ + _, TripletFields.All).mapValues { z =>
      val q = 1.0 / (1.0 + exp(z))
      // if (q.isInfinite || q.isNaN || q == 0.0) println(z)
      assert(q != 0.0)
      q
    }
  }

  def backward(q: VertexRDD[VD]): VertexRDD[Double] = {
    dataSet.outerJoinVertices(q) { (_, label, qv) =>
      (label, qv.getOrElse(0.0))
    }.aggregateMessages[Array[Double]](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val y = ctx.dstAttr._1
      val q = ctx.dstAttr._2 * abs(x)
      assert(q != 0.0)
      val mu = if (signum(x * y) > 0.0) {
        Array(q, 0.0)
      } else {
        Array(0.0, q)
      }
      ctx.sendToSrc(mu)
    }, (a, b) => Array(a(0) + b(0), a(1) + b(1)), TripletFields.Dst).mapValues { mu =>
      if (mu.min == 0.0) 0.0 else math.log((mu(0) + epsilon) / (mu(1) + epsilon))
    }
  }

  // Updater for L1 regularized problems
  def updateWeight(delta: VertexRDD[Double], iter: Int): Graph[VD, ED] = {
    // val thisIterStepSize = stepSize / sqrt(iter)
    val thisIterStepSize = stepSize
    dataSet.outerJoinVertices(delta) { (_, attr, mu) =>
      mu match {
        case Some(gard) => {
          var weight = attr
          weight = weight + thisIterStepSize * gard
          if (regParam > 0.0) {
            val shrinkageVal = regParam * thisIterStepSize
            weight = signum(weight) * max(0.0, abs(weight) - shrinkageVal)
          }
          assert(!weight.isNaN)
          weight
        }
        case None => attr
      }
    }
  }

  private def checkpoint(corpus: Graph[VD, ED]): Graph[VD, ED] = {
    if (innerIter % 10 == 0 && corpus.edges.sparkContext.getCheckpointDir.isDefined) {
      logInfo(s"start checkpoint")
      corpus.checkpoint()
      val newVertices = corpus.vertices.mapValues(t => t)
      val newCorpus = GraphImpl(newVertices, corpus.edges)
      newCorpus.checkpoint()
      logInfo(s"end checkpoint")
      newCorpus
    } else {
      corpus
    }
  }
}

object LRonGraphX {
  private[mllib] type ED = Double
  private[mllib] type VD = Double

  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    stepSize: Double,
    regParam: Double): LogisticRegressionModel = {
    train(input, numIterations, 0, stepSize, regParam)
  }

  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    numFeatures: Int,
    stepSize: Double,
    regParam: Double,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): LogisticRegressionModel = {
    val data = input.zipWithIndex().map { case (LabeledPoint(label, features), id) =>
      features.foreachActive((index, value) => assert(abs(value) <= 1.0))
      val newLabel = if (label > 0.0) 1.0 else -1.0
      (id, LabeledPoint(newLabel, features))
    }
    val newNumFeatures = if (numFeatures < 1) {
      data.first()._2.features.size
    } else {
      numFeatures
    }
    val lr = new LRonGraphX(data, newNumFeatures, stepSize, regParam, 0.0, storageLevel)
    lr.run(numIterations)
    val model = lr.saveModel()
    data.unpersist()
    model
  }

  private def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.flatMap { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      labelPoint.features.toBreeze.activeIterator.map { case (index, value) =>
        Edge(index, newId, value)
      }
    }
    val vertices = input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      (newId, labelPoint.label)
    }
    var dataSet = Graph.fromEdges(edges, null, storageLevel, storageLevel)

    // degree-based hashing
    val numPartitions = edges.partitions.size
    val partitionStrategy = new DBHPartitioner(numPartitions)
    val newEdges = dataSet.outerJoinVertices(dataSet.degrees) { (vid, data, deg) =>
      deg.getOrElse(0)
    }.triplets.map { e =>
      (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    dataSet = Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
    // end degree-based hashing
    // dataSet = dataSet.partitionBy(PartitionStrategy.EdgePartition2D)

    dataSet.outerJoinVertices(vertices) { (vid, data, deg) =>
      // deg.getOrElse(Utils.random.nextGaussian() * 1e-2)
      deg.getOrElse(0)
    }
  }

  private def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }
}


/**
 * Degree-Based Hashing, the paper:
 * Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
 */
private class DBHPartitioner(val partitions: Int) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  def numPartitions = partitions

  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
    val srcDeg = edge.srcAttr
    val dstDeg = edge.dstAttr
    val srcId = edge.srcId
    val dstId = edge.dstId
    if (srcDeg < dstDeg) {
      getPartition(srcId)
    } else {
      getPartition(dstId)
    }
  }

  def getPartition(idx: Int): PartitionID = {
    getPartition(idx.toLong)
  }

  def getPartition(idx: Long): PartitionID = {
    (abs(idx * mixingPrime) % partitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case h: DBHPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}
