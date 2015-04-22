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
import breeze.linalg.{norm => brzNorm, argmax => brzArgMax}
import breeze.linalg.functions.euclideanDistance

class SentenceClassifierSuite extends FunSuite with MLlibTestSparkContext {

  test("SentenceClassifier") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    import org.apache.spark.mllib.feature._
    import breeze.linalg.{norm => brzNorm}
    // http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label
    val txt = sc.textFile(s"$sparkHome/data/mllib/sst/train_5500.label").
      map { line =>
      line.split(":").map(_.split(" ")).flatten.map(_.trim).filter(_.nonEmpty)
    }.filter(_.length > 1).map(_.toIterable).cache()
    println("txt " + txt.count)
    val word2Vec = new Word2Vec()
    word2Vec.setVectorSize(64).setNumIterations(3)
    val model = word2Vec.fit(txt.map(_.tail))
    val Array(txtTrain, txtTest) = txt.repartition(2).randomSplit(Array(0.7, 0.3))
    val (sent2vec, wordVec, wordIndex, labelIndex) =
      SentenceClassifier.train(txtTrain.cache(), model, 1000, 0.01, 0.02)
    println(s"wordVec ${wordVec.valuesIterator.map(_.abs).sum / wordVec.length}")

    val vecs = txtTest.map { t =>
      val vec = t.tail.filter(w => wordIndex.contains(w)).map(w => wordIndex(w)).toArray
      (t, vec)
    }.filter(_._2.length > 4).map { sent =>
      sent2vec.setWord2Vec(wordVec)
      val label = brzArgMax(sent2vec.predict(sent._2))
      (sent._1.head, label)
    }.cache()

    val sum = vecs.count
    val err = vecs.filter(t => labelIndex(t._1) != t._2).count
    println(s"Error: $err / $sum = ${err.toDouble / sum * 100} % ")

  }
}
