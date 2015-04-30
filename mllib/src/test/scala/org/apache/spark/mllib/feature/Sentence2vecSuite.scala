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

package org.apache.spark.mllib.feature

import breeze.linalg.functions.euclideanDistance
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite

class Sentence2vecSuite extends FunSuite with MLlibTestSparkContext {
  test("Sentence2vec docs") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    // http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label
    val txt = sc.textFile(s"$sparkHome/data/mllib/sst/train_5500.label").
      map(line => line.split(":").last.split(" ").tail).filter(_.length > 4).map(_.toIterable).cache()
    println("txt " + txt.count)
    val (sent2vec, word2, word2Index) = Sentence2vec.train(txt, 64, 1000, 0.05, 0.01)
    println(s"word2 ${word2.valuesIterator.map(_.abs).sum / word2.length}")
    val vecs = txt.map { t =>
      val vec = t.filter(w => word2Index.contains(w)).map(w => word2Index(w)).toArray
      (t, vec)
    }.filter(_._2.length > 4).map { sent =>
      sent2vec.setWord2Vec(word2)
      val vec = sent2vec.predict(sent._2)
      (sent._1, vec)
    }.cache()
    vecs.takeSample(false, 10).foreach { case (text, vec) =>
      println(s"${text.mkString(" ")}")
      vecs.map(v => {
        val sim: Double = euclideanDistance(v._2, vec)
        (sim, v._1)
      }).sortByKey(true).take(4).foreach(t => println(s"${t._1} =>${t._2.mkString(" ")} \n"))
    }

  }
  ignore("dealInfo") {
    import breeze.linalg.functions._
    import org.apache.spark.mllib.feature._
    val deal2vecPath = "/input/lbs/recommend/toona/deal2vec/"
    val deals = sc.textFile(deal2vecPath).map { line =>
      line.split(" ")
    }.filter(_.length > 3).map(_.toIterable).repartition(72).persist()
    deals.count()
    val (sent2vec, word2, word2Index) = Sentence2vec.train(deals, 100, 4000, 0.05, 2e-3)
    println(s"word2 ${word2.valuesIterator.map(_.abs).sum / word2.length}")
    val vecs = deals.map { t =>
      val vec = t.filter(w => word2Index.contains(w)).map(w => word2Index(w)).toArray
      (t, vec)
    }.filter(_._2.length > 4).map { sent =>
      sent2vec.setWord2Vec(word2)
      val vec = sent2vec.predict(sent._2)
      (sent._1, vec)
    }.cache()
    vecs.takeSample(false, 10).foreach { case (text, vec) =>
      println(s"${text.mkString(" ")}")
      vecs.map(v => {
        val sim: Double = euclideanDistance(v._2, vec)
        (sim, v._1)
      }).filter(_._1 != 0.0).sortByKey(true).take(6).foreach(t => println(s"${t._1} =>${t._2.mkString(" ")} \n"))
    }
  }
}
