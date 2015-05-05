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

import java.util.{PriorityQueue => JPriorityQueue, Random}

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm => brzNorm, sum => brzSum}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV}

private[mllib] case class AliasTable(var l: Array[Int], var h: Array[Int],
  var p: Array[Double], var used: Int) {
  def sampleAlias(gen: Random): Int = {
    val bin = gen.nextInt(used)
    val prob = p(bin)
    if (used * prob > gen.nextDouble()) {
      l(bin)
    } else {
      h(bin)
    }
  }
}

private[mllib] object AliasTable {
  @transient private lazy val tableOrdering = new scala.math.Ordering[(Int, Double)] {
    override def compare(x: (Int, Double), y: (Int, Double)): Int = {
      Ordering.Double.compare(x._2, y._2)
    }
  }
  @transient private lazy val tableReverseOrdering = tableOrdering.reverse

  def generateAlias(sv: BV[Double]): AliasTable = {
    val used = sv.activeSize
    val sum = brzSum(sv)
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(sv: BV[Double], sum: Double): AliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(sv: BV[Double], sum: Double, table: AliasTable): AliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used, table)
  }

  def generateAlias(
    probs: Iterator[(Int, Double)],
    sum: Double,
    used: Int): AliasTable = {
    val table = AliasTable(new Array[Int](used), new Array[Int](used), new Array[Double](used), used)
    generateAlias(probs, sum, used, table)
  }

  def generateAlias(
    probs: Iterator[(Int, Double)],
    sum: Double,
    used: Int,
    table: AliasTable): AliasTable = {
    table.used = used
    val pMean = 1.0 / used
    val lq = new JPriorityQueue[(Int, Double)](used, tableOrdering)
    val hq = new JPriorityQueue[(Int, Double)](used, tableReverseOrdering)

    probs.slice(0, used).foreach { pair =>
      val i = pair._1
      val pi = pair._2 / sum
      if (pi < pMean) {
        lq.add((i, pi))
      } else {
        hq.add((i, pi))
      }
    }

    var offset = 0
    while (!lq.isEmpty & !hq.isEmpty) {
      val (i, pi) = lq.remove()
      val (h, ph) = hq.remove()
      table.l(offset) = i
      table.h(offset) = h
      table.p(offset) = pi
      val pd = ph - (pMean - pi)
      if (pd >= pMean) {
        hq.add((h, pd))
      } else {
        lq.add((h, pd))
      }
      offset += 1
    }
    while (!hq.isEmpty) {
      val (h, ph) = hq.remove()
      assert(ph - pMean < 1e-8)
      table.l(offset) = h
      table.h(offset) = h
      table.p(offset) = ph
      offset += 1
    }

    while (!lq.isEmpty) {
      val (i, pi) = lq.remove()
      assert(pMean - pi < 1e-8)
      table.l(offset) = i
      table.h(offset) = i
      table.p(offset) = pi
      offset += 1
    }
    table
  }

}