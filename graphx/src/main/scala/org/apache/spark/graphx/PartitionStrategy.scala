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

package org.apache.spark.graphx

import scala.util.Random

/**
 * Represents the way edges are assigned to edge partitions based on their source and destination
 * vertex IDs.
 */
trait PartitionStrategy extends Serializable {
  /** Returns the partition number for a given edge. */
  def getPartition(src: VertexId, dst: VertexId, numParts: PartitionID): PartitionID
}

/**
 * Collection of built-in [[PartitionStrategy]] implementations.
 */
object PartitionStrategy {

  /**
   * Assigns edges to partitions using a 2D partitioning of the sparse edge adjacency matrix,
   * guaranteeing a `2 * sqrt(numParts)` bound on vertex replication.
   *
   * Suppose we have a graph with 11 vertices that we want to partition
   * over 9 machines.  We can use the following sparse matrix representation:
   *
   * <pre>
   * __________________________________
   * v0   | P0 *     | P1       | P2    *  |
   * v1   |  ****    |  *       |          |
   * v2   |  ******* |      **  |  ****    |
   * v3   |  *****   |  *  *    |       *  |
   * ----------------------------------
   * v4   | P3 *     | P4 ***   | P5 **  * |
   * v5   |  *  *    |  *       |          |
   * v6   |       *  |      **  |  ****    |
   * v7   |  * * *   |  *  *    |       *  |
   * ----------------------------------
   * v8   | P6   *   | P7    *  | P8  *   *|
   * v9   |     *    |  *    *  |          |
   * v10  |       *  |      **  |  *  *    |
   * v11  | * <-E    |  ***     |       ** |
   * ----------------------------------
   * </pre>
   *
   * The edge denoted by `E` connects `v11` with `v1` and is assigned to processor `P6`. To get the
   * processor number we divide the matrix into `sqrt(numParts)` by `sqrt(numParts)` blocks. Notice
   * that edges adjacent to `v11` can only be in the first column of blocks `(P0, P3,
   * P6)` or the last
   * row of blocks `(P6, P7, P8)`.  As a consequence we can guarantee that `v11` will need to be
   * replicated to at most `2 * sqrt(numParts)` machines.
   *
   * Notice that `P0` has many edges and as a consequence this partitioning would lead to poor work
   * balance.  To improve balance we first multiply each vertex id by a large prime to shuffle the
   * vertex locations.
   *
   * One of the limitations of this approach is that the number of machines must either be a
   * perfect square. We partially address this limitation by computing the machine assignment to
   * the next
   * largest perfect square and then mapping back down to the actual number of machines.
   * Unfortunately, this can also lead to work imbalance and so it is suggested that a perfect
   * square is used.
   */
  case object EdgePartition2D extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val ceilSqrtNumParts: PartitionID = math.ceil(math.sqrt(numParts)).toInt
      val mixingPrime: VertexId = 1125899906842597L
      val col: PartitionID = (math.abs(src * mixingPrime) % ceilSqrtNumParts).toInt
      val row: PartitionID = (math.abs(dst * mixingPrime) % ceilSqrtNumParts).toInt
      ((col * ceilSqrtNumParts + row) % numParts).toInt
    }
  }

  // solve square => factor
  case object EdgePartition2DV1 extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val ceilSqrtNumParts: PartitionID =
        math.ceil(math.sqrt(numParts)).toInt * math.ceil(math.sqrt(numParts)).toInt
      val mixingPrime: VertexId = 1125899906842597L
      val col: PartitionID = (math.abs(src * mixingPrime) % ceilSqrtNumParts).toInt
      val row: PartitionID = (math.abs(dst * mixingPrime) % ceilSqrtNumParts).toInt
      ((col * ceilSqrtNumParts + row) % numParts).toInt
    }
  }

  // solve square => rectangle
  case object EdgePartition2DV2 extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val ceilSqrtNumParts: PartitionID = math.ceil(math.sqrt(numParts)).toInt
      val mixingPrime: VertexId = 1125899906842597L
      val col: PartitionID = (math.abs(src * mixingPrime) % ceilSqrtNumParts).toInt
      val row: PartitionID = (math.abs(dst * mixingPrime) % ceilSqrtNumParts).toInt
      ((col * ceilSqrtNumParts + row) % numParts).toInt
    }
  }

  /* Canonical
  O---
  OO--
  OOO-
  OOOO
  1 3 6 10 15 21 28 36 45 55 66 78 91
  */
  case object CanonicalEdgePartition2DV2 extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val ceilSqrtNumParts: PartitionID =
        math.ceil(math.sqrt(numParts)).toInt * 10
      val mixingPrime: VertexId = 1125899906842597L
      val col: PartitionID = (math.abs(src * mixingPrime) % ceilSqrtNumParts).toInt
      val row: PartitionID = (math.abs(dst * mixingPrime) % ceilSqrtNumParts).toInt
      if (col >= row) {
        ((col * ceilSqrtNumParts + row) % numParts).toInt
      } else {
        ((row * ceilSqrtNumParts + col) % numParts).toInt
      }
    }
  }

  /* Grid
  OO*O
  OOOO
  *OOO
  OOOO
  */
  case object CanonicalEdgePartition2DV1 extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val ceilSqrtNumParts: PartitionID = math.ceil(math.sqrt(numParts)).toInt
      val mixingPrime: VertexId = 1125899906842597L
      val col: PartitionID = (math.abs(src * mixingPrime) % ceilSqrtNumParts).toInt
      val row: PartitionID = (math.abs(dst * mixingPrime) % ceilSqrtNumParts).toInt
      val rand = new Random()
      //random true / false
      if (rand.nextBoolean()) {
        ((col * ceilSqrtNumParts + row) % numParts).toInt
      } else {
        ((row * ceilSqrtNumParts + col) % numParts).toInt
      }

    }
  }

  /**
   * Assigns edges to partitions using only the source vertex ID, colocating edges with the same
   * source.
   */
  case object EdgePartition1DSrc extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val mixingPrime: VertexId = 1125899906842597L
      (math.abs(src * mixingPrime) % numParts).toInt
    }
  }

  /**
   * Assigns edges to partitions using only the destination vertex ID, colocating edges with the same
   * destination.
   */
  case object EdgePartition1DDst extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      val mixingPrime: VertexId = 1125899906842597L
      ((math.abs(dst) * mixingPrime) % numParts).toInt
    }
  }


  /**
   * Assigns edges to partitions by hashing the source and destination vertex IDs, resulting in a
   * random vertex cut that colocates all same-direction edges between two vertices.
   */
  case object RandomVertexCut extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      (math.abs((src, dst).hashCode()) % numParts).toInt
    }
  }


  /**
   * Assigns edges to partitions by hashing the source and destination vertex IDs in a canonical
   * direction, resulting in a random vertex cut that colocates all edges between two vertices,
   * regardless of direction.
   */
  case object CanonicalRandomVertexCut extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      if (src < dst) {
        (math.abs((src, dst).hashCode()) % numParts).toInt
      } else {
        (math.abs((dst, src).hashCode()) % numParts).toInt
      }
    }
  }

  /**
   * @brief HybridCut, a PartitionStrategy
   * @details 
   *
   * a Balanced p-way Hybrid-Cut inspired by
   * PowerLyra : Differentiated Graph Computation and Partitioning on Skewed Graphs 
   * Institute of Parallel and Distributed Systems 
   * Chen, R., Shi, J., Chen, Y., Guan, H., Chen, H., Ipadstr--, T. R. N., & Zang, B. (2013). 
   *
   * design 1 (same parameter as before)
   * @param src [source vertex ID]
   * @param dst [destination vertex ID]
   * @param numParts [number of partitions intended to part]
   * @return PartitionID [PartitionID for this edge]
   *         logic: need graph info
   */
  case object GreedyHybridCut extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      0
    }
  }

  case object HybridCutPlus extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      0
    }
  }

  case object HybridCut extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      // this is how to get in-degrees for a given graph
      // need to persist it and for store reference use
      // make it one copy, destroy when finish?

      // info Vertex ID or Vetex Name?
      // val inDegrees: VertexRDD[Int] = graph.inDegrees
      // var DegreesArray = inDegrees.toArray()
      // var DegreesMap = Map(DegreesArray:_*)
      // DegreesMap.get(src)
      // var ThreshHold = 50
      // // val DegreeCount : Int = DegreesMap.getOrElse(dst, 0)
      // val DegreeCount : Int = inDegrees.lookup(dst).head
      // if (DegreeCount > ThreshHold) {
      //     // high-cut
      //     math.abs(src).toInt % numParts
      //   } else {
      //     // low-cut
      //     math.abs(dst).toInt % numParts
      //   }
      // dummy
      (math.abs(dst) % numParts).toInt
    }
  }

  case object BiSrcCut extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      (math.abs(dst) % numParts).toInt
    }
  }

  case object BiDstCut extends PartitionStrategy {
    override def getPartition(
      src: VertexId, dst: VertexId, numParts: PartitionID)
    : PartitionID = {
      (math.abs(dst) % numParts).toInt
    }
  }

  /** Returns the PartitionStrategy with the specified name. */
  def fromString(s: String): PartitionStrategy = s match {
    case "RandomVertexCut" => RandomVertexCut
    case "EdgePartition1DSrc" => EdgePartition1DSrc
    case "EdgePartition1DDst" => EdgePartition1DDst
    case "EdgePartition2D" => EdgePartition2D
    case "CanonicalEdgePartition2DV1" => CanonicalEdgePartition2DV1
    case "CanonicalEdgePartition2DV2" => CanonicalEdgePartition2DV2
    case "EdgePartition2DV1" => EdgePartition2DV1
    case "EdgePartition2DV2" => EdgePartition2DV2
    case "CanonicalRandomVertexCut" => CanonicalRandomVertexCut
    case "HybridCut" => HybridCut
    case "HybridCutPlus" => HybridCutPlus
    case "GreedyHybridCut" => GreedyHybridCut
    case "BiSrcCut" => BiSrcCut
    case "BiDstCut" => BiDstCut
    case _ => throw new IllegalArgumentException("Invalid PartitionStrategy: " + s)
  }

}
