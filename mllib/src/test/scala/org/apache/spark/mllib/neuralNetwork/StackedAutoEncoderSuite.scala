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

package org.apache.spark.mllib.neuralNetwork

import org.scalatest.{Matchers, FunSuite}
import org.apache.spark.mllib.util.MnistDatasetSuite
import breeze.linalg.{SparseVector => BSV}
import org.apache.spark.mllib.linalg.{SparseVector => SSV, Vector => SV}

class StackedAutoEncoderSuite extends FunSuite
with MnistDatasetSuite with Matchers {
  test("StackedAutoEncoder") {
    val dataPtah = "/Users/witgo/work/code/java/spark/data/mllib/data.txt"
    val txt = sc.textFile(dataPtah).map(t => t.split(" ")).sample(false, 0.2)
    val dataFeatures = txt.zipWithIndex.map(_.swap)
    dataFeatures.count()
    val index = dataFeatures.map(_._2).flatMap(t => t).distinct().
      zipWithIndex.map(t => (t._1, t._2.toInt)).cache()
    val numTerms = index.count.toInt
    println(numTerms)
    val docs = dataFeatures.flatMap {
      t => t._2.map(c => (c, t._1))
    }.join(index).map { case (word, (docId, termId)) =>
      (docId, termId)
    }.groupByKey().map { case (docID, features) =>
      val v = BSV.zeros[Double](numTerms)
      features.foreach(term => v(term) += 1)
      val sv = new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
      sv.asInstanceOf[SV]
    }
    val dbn = new StackedAutoEncoder(Array(numTerms, 10, numTerms))
    // StackedAutoEncoder.pretrain(docs, 1, 1000, dbn, 0.001, 0.1, 0.0)
    StackedAutoEncoder.finetune(docs, 1, 8000, dbn, 0.0005, 0.05, 0.0)
  }
}
