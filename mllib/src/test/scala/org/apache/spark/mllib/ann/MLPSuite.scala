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

package org.apache.spark.mllib.ann

import org.apache.spark.mllib.util.MnistDatasetSuite
import org.scalatest.{FunSuite, Matchers}

class MLPSuite extends FunSuite with MnistDatasetSuite with Matchers {
  ignore("mnist LBFGS") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val hiddenLayersTopology = Array[Int](20)
    val layerSizes = numVisible +: hiddenLayersTopology :+ 10
    val topology = Topology.multiLayerPerceptron(layerSizes, true)
    val initialWeights = FeedForwardModel(topology, 117).weights()
    val trainer = new FeedForwardTrainer(topology, numVisible, 10)
    trainer.LBFGSOptimizer.setNumIterations(100)
    trainer.setWeights(initialWeights)
    val model = trainer.train(data)

  }

  test("mnist SGD") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val hiddenLayersTopology = Array[Int](20)
    val layerSizes = numVisible +: hiddenLayersTopology :+ 10
    val topology = Topology.multiLayerPerceptron(layerSizes, true)
    val initialWeights = FeedForwardModel(topology, 117).weights()
    val trainer = new FeedForwardTrainer(topology, numVisible, 10)
    trainer.SGDOptimizer.setNumIterations(2000).setMiniBatchFraction(1).setStepSize(0.1)
    trainer.setWeights(initialWeights)
    val model = trainer.train(data)

  }
}
