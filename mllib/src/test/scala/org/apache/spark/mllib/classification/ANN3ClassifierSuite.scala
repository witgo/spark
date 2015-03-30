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

import org.apache.spark.mllib.ann2.{FeedForwardTrainer, FeedForwardModel, Topology}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite

class ANN3ClassifierSuite extends FunSuite with MLlibTestSparkContext {

  test("ANN classifier test for XOR"){
    val inputs = Array[Array[Double]](
      Array[Double](0,0),
      Array[Double](0,1),
      Array[Double](1,0),
      Array[Double](1,1)
    )
    val outputs = Array[Double](0, 1, 1, 0)
    val data = inputs.zip(outputs).map{ case(input, output) =>
      new LabeledPoint(output, Vectors.dense(input))}
    val rddData = sc.parallelize(data, 2)
    val layers = Array[Int](2, 5, 2)
    val topology = Topology.multiLayerPerceptron(layers)
    val initialWeights = FeedForwardModel(topology, 0x01234567).weights()
    val trainer = new FeedForwardTrainer(topology, 2, 2)
    trainer.setWeights(initialWeights).setBatchSize(1)
    trainer.LBFGSOptimizer.setNumIterations(200).setConvergenceTol(1e-4)
    val model = new ANN3Classifier(trainer).train(rddData)
    //val initialWeights = ANNClassifier.randomWeights(rddData, hiddenLayerTopology, 0x01234567)
    //val model = ANNClassifier.train(rddData, 1, hiddenLayerTopology, initialWeights, 200, 1.0, 1e-4)
    val predictionAndLabels = rddData.map(lp =>
      (model.predict(lp.features), lp.label)).collect()
    assert(predictionAndLabels.forall { case(p, l) =>
      (p - l) == 0 })
  }
}
