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

import java.util
import java.util.{PriorityQueue => JPriorityQueue}

import scala.math._
import scala.collection.JavaConversions._

import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.{HashPartitioner, Logging, Partitioner}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.AppendOnlyMap
import org.apache.spark.util.Utils


import LRonGraphX._

class LRonGraphX(
  @transient var dataSet: Graph[VD, ED],
  val numFeatures: Int,
  val stepSize: Double = 1e-4,
  val regParam: Double = 1e-2,
  val epsilon: Double = 1e-6,
  @transient var storageLevel: StorageLevel =
  StorageLevel.MEMORY_AND_DISK) extends Serializable with Logging {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    numFeatures: Int,
    stepSize: Double,
    regParam: Double) {
    this(initializeDataSet(input, StorageLevel.MEMORY_AND_DISK), numFeatures, stepSize, regParam)
  }

  @transient private var innerIter = 1
  lazy val numSamples: Long = samples.count()

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 < 0)
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 >= 0)
  }

  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      val previousDataSet = dataSet
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val q = forward()
      logInfo(s"train (Iteration $iter/$iterations) MSE: ${meanSquaredError(q)}")
      val delta = backward(q)
      dataSet = updateWeight(delta, iter)
      dataSet = checkpoint(dataSet)
      dataSet.persist(storageLevel)
      dataSet.vertices.count()
      dataSet.edges.count()
      previousDataSet.unpersist()
      logInfo(s"End train (Iteration $iter/$iterations)")
      innerIter += 1
    }
  }

  def saveModel(): LogisticRegressionModel = {
    val featureData = new Array[VD](numFeatures)
    features.toLocalIterator.foreach { case (index, value) =>
      featureData(index.toInt) = value
    }
    new LogisticRegressionModel(new SDV(featureData), 0.0)
  }

  def meanSquaredError(q: VertexRDD[VD]): Double = {
    samples.join(q).map { case (_, (label, score)) =>
      val diff = max(label, 0.0) - score
      pow(diff, 2)
    }.sum / numSamples
  }

  def forward(): VertexRDD[VD] = {
    dataSet.aggregateMessages[Double](ctx => {
      // val sampleId = ctx.dstId
      // val featureId = ctx.srcId
      val x = ctx.attr
      val w = ctx.srcAttr
      val label = ctx.dstAttr
      val v = signum(label) * w * x
      assert(!v.isNaN)
      ctx.sendToDst(v)
    }, _ + _, TripletFields.All).mapValues { v =>
      val q = 1.0 / (1.0 + Math.exp(v))
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
      val label = ctx.dstAttr._1
      val qVal = ctx.dstAttr._2
      assert(qVal != 0.0)
      val mu = if (label > 0.0) {
        Array(qVal, 0.0)
      } else {
        Array(0.0, qVal)
      }
      ctx.sendToSrc(mu)
    }, (a, b) => Array(a(0) + b(0), a(1) + b(1)), TripletFields.Dst).mapValues { mu =>
      if (mu.min == 0.0) 0.0 else math.log(mu(0) / mu(1))
      // math.log((mu(0) + epsilon) / (mu(1) + epsilon))
    }
  }

  // Updater for L1 regularized problems
  def updateWeight(delta: VertexRDD[Double], iter: Int): Graph[VD, ED] = {
    val thisIterStepSize = stepSize / sqrt(iter)
    dataSet.outerJoinVertices(delta) { (_, attr, mu) =>
      mu match {
        case Some(gard) => {
          var weight = attr
          weight += thisIterStepSize * gard
          val shrinkageVal = regParam * thisIterStepSize
          weight = signum(weight) * max(0.0, abs(weight) - shrinkageVal)
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
    val data = input.zipWithIndex().map(_.swap).map { case (id, LabeledPoint(label, features)) =>
      val values = features match {
        case SDV(values) => values
        case SSV(size, indices, values) => values
      }

      //  val sum = values.map(abs).sum
      //  for (i <- 0 until values.length) {
      //    values(i) = values(i) / sum
      //  }

      for (i <- 0 until values.length) {
        val v = values(i)
        values(i) = max(signum(v), 0)
      }
      val newLabel = if (label > 0.0) 1.0 else -1.0
      (id, LabeledPoint(newLabel, features))
    }
    data.persist(storageLevel)
    val dataSet = initializeDataSet(data, storageLevel)
    val lr = new LRonGraphX(dataSet, if (numFeatures < 1) {
      data.first()._2.features.size
    } else {
      numFeatures
    }, stepSize, regParam)
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
    val vertices = VertexRDD(input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      (newId, labelPoint.label)
    })
    var dataSet = Graph.fromEdges(edges, null)

    // degree-based hashing
    //  val degrees = dataSet.degrees
    //  val threshold = (degrees.map(_._2.toDouble).sum() /
    //    (2.0 * degrees.filter(t => t._1 < 0).count())).ceil.toInt
    //  println(s"threshold $threshold")
    //  val numPartitions = edges.partitions.size
    //  val partitionStrategy = new DBHPartitioner(numPartitions, threshold)
    //  val newEdges = dataSet.triplets.map { e =>
    //    (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
    //  }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    //  dataSet = Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
    // end degree-based hashing

    // dynamic hashing
    val numPartitions = edges.partitions.size
    val degrees = dataSet.outerJoinVertices(dataSet.degrees) { (vid, data, deg) =>
      deg.getOrElse(0)
    }
    val newEdges = degrees.triplets.mapPartitions { iter =>
      val partitionStrategy = new DynamicPartitioner(numPartitions)
      iter.map { e => (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr)) }
    }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
    dataSet = Graph.fromEdges(newEdges, null, storageLevel, storageLevel)

    println(s"edges distribution($numPartitions)")
    dataSet.edges.mapPartitions(t => Iterator(t.size)).toLocalIterator.foreach(println)
    // end dynamic hashing


    dataSet.outerJoinVertices(vertices) { (vid, data, deg) =>
      deg.getOrElse(Utils.random.nextGaussian() * 1e-2)
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
private class DBHPartitioner(val partitions: Int, val threshold: Int = 70) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  def numPartitions = partitions

  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
    val idMin = min(edge.srcAttr, edge.dstAttr)
    val idMax = max(edge.srcAttr, edge.dstAttr)
    if (idMax < threshold) {
      getPartition(idMax)
    } else {
      getPartition(idMin)
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


private class DynamicPartitioner(val numPartitions: Int) extends Serializable {
  @transient lazy val mixingPrime: Long = 1125899906842597L
  @transient lazy val vert2Pid = new AppendOnlyMap[VertexId, Int]()
  @transient lazy val pid2Size = new AppendOnlyMap[Int, Long]()

  for (i <- 1 to numPartitions) {
    pid2Size(i) = 0
  }

  def getPartition(edge: EdgeTriplet[Int, ED]): Int = {
    val srcId = edge.srcId
    val dstId = edge.dstId
    val srcDeg = edge.srcAttr
    val dstDeg = edge.dstAttr
    val srcPid = vert2Pid(srcId)
    val dstPid = vert2Pid(dstId)

    val edgePid = if (srcPid > 0 && dstPid > 0) {
      val degPid = if (srcDeg <= dstDeg) srcPid else dstPid
      updatePid(degPid)
      degPid
    } else if (srcPid > 0 || dstPid > 0) {
      val pid = if (srcPid > 0) srcPid else dstPid
      val vertId = if (srcPid > 0) dstId else srcId
      vert2Pid.update(vertId, pid)
      updatePid(pid)
      pid
    } else {
      val pid = minSizePid
      vert2Pid.update(srcId, pid)
      vert2Pid.update(dstId, pid)
      updatePid(pid)
      pid
    }
    edgePid - 1
  }

  def updatePid(pid: Int): Long = {
    pid2Size.changeValue(pid, (_, oldValue) => oldValue + 1L)
  }

  def minSizePid: Int = {
    val itr = pid2Size.iterator
    var pid = 1
    var pidSize = Long.MaxValue
    while (itr.hasNext) {
      val t = itr.next()
      val tPid = t._1
      val tSize = t._2
      if (tSize < pidSize) {
        pid = tPid
        pidSize = tSize
      }
    }
    pid
  }


  def getPartition(idx: Int): PartitionID = {
    getPartition(idx.toLong)
  }

  def getPartition(idx: Long): PartitionID = {
    (abs(idx * mixingPrime) % numPartitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case h: DynamicPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}
