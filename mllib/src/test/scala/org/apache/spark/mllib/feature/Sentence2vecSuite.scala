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

class Sentence2vecSuite extends FunSuite with MLlibTestSparkContext {

  test("Sentence2vec") {
    import org.apache.spark.mllib.feature._
    val txt = sc.textFile("/Users/witgo/work/code/java/spark/data/mllib/trainings.txt").
      map(_.split(" ").
      filter(w => w.length > 2).
      filter(w => !w.contains("_")).
      filter(w => !w.contains("-"))).
      filter(_.length > 4).
      map(_.toIterable)
    val word2Vec = new Word2Vec()
    word2Vec.
      setVectorSize(10).
      setSeed(42L).
      setNumIterations(5).
      setNumPartitions(2)
    val model = word2Vec.fit(txt)

    Sentence2vec.train(txt, model, 10000, 0.1)

  }
}
