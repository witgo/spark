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

import org.scalatest.FunSuite

import org.apache.spark.mllib.util.MLlibTestSparkContext

import org.apache.spark.SparkContext._
import breeze.linalg.{norm => brzNorm}

class Sentence2vecSuite extends FunSuite with MLlibTestSparkContext {

  test("Sentence2vec") {
    import org.apache.spark.mllib.feature._
    val txt = sc.textFile("/Users/witgo/work/code/java/spark/data/mllib/data.txt").
      map(_.split(" ").
      filter(w => w.length > 1).
      filter(w => !w.contains("_")).
      filter(w => !w.contains("-"))).
      filter(_.length > 4).sample(false, 0.5).
      map(_.toIterable).cache()
    println("txt " + txt.count)
    val word2Vec = new Word2Vec()
    word2Vec.
      setVectorSize(100).
      setSeed(12324L)
    val model = word2Vec.fit(txt)
    val (sent2vec, word2, word2Index) = Sentence2vec.train(txt, model, 500, 0.05, 0.0005)
    val vecs = txt.map { t =>
      val vec = t.filter(w => word2Index.contains(w)).map(w => word2Index(w)).toArray
      (t, vec)
    }.filter(_._2.length > 4).map { sent =>
      sent2vec.setWord2Vec(word2)
      val vec = sent2vec.predict(sent._2)
      vec :/= brzNorm(vec, 2.0)
      (sent._1, vec)
    }.cache()
    vecs.takeSample(false, 10).foreach { case (text, vec) =>
      println(s"${text.mkString(" ")}")
      vecs.map(v => {
        val sim: Double = v._2.dot(vec)
        (sim, v._1)
      }).sortByKey(false).take(4).foreach(t => println(s"${t._1} =>${t._2.mkString(" ")} \n"))
    }
  }
}
