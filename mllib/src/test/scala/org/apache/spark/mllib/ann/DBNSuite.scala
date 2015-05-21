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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV,
argmax => brzArgMax, axpy => brzAxpy, max => brzMax, norm => brzNorm, sum => brzSum}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MnistDatasetSuite
import org.apache.spark.rdd.RDD
import org.scalatest.{FunSuite, Matchers}

class DBNSuite extends FunSuite with MnistDatasetSuite with Matchers {
  ignore("mnist SGD") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val numOut = 10
    val topology = RBMTopology(numVisible, numOut, 0)
    val initialWeights = topology.getInstance(117).weights()
    val trainer = new RBMTrainer(topology)
    trainer.SGDOptimizer.setNumIterations(2000).setMiniBatchFraction(1).setStepSize(0.01)
    // val updater = new AdaGradUpdater(0, 1e-6, 0.9)
    val updater = new EquilibratedUpdater(1e-6, 0)
    trainer.setWeights(initialWeights).setUpdater(updater)
    // trainer.setWeights(initialWeights)
    val model = trainer.train(data.map(_._1))
  }

  ignore("StackedRBM") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val numOut = 10
    val topology = StackedRBMTopology.multiLayer(Array(numVisible, 100, 500))
    val trainer = new StackedRBMTrainer(topology).
      setNumIterations(50).
      setMiniBatchFraction(1).
      setStepSize(0.01)
    val model = trainer.train(data.map(_._1))
  }

  test("DBN") {
    val (data, numVisible) = mnistTrainDataset(5000)
    val numOut = 10
    val topology = StackedRBMTopology.multiLayer(Array(numVisible, 500))
    val topTopology = FeedForwardTopology.multiLayerPerceptron(Array(500, numOut), true)

    val trainer = new DBNTrainer(topology, topTopology).
      setNumIterations(500).
      setMiniBatchFraction(0.02).
      setStepSize(0.005).
      setBatchSize(20).
      setRegParam(5e-4)

    val model = trainer.pretrain(data)
    val ann = trainer.finetune(data, model)
    // val ann = trainer.finetune(data)

    val (dataTest, _) = mnistTrainDataset(5000, 5000)
    println(f"Accuracy: ${1 - error(dataTest, ann)}%1.6f")
  }

  def error(data: RDD[(Vector, Vector)], nn: TopologyModel): Double = {
    val count = data.count()

    val sumError = data.map { case (x, y) =>
      val h = nn.predict(x)
      if (brzArgMax(y.toBreeze.asInstanceOf[BDV[Double]]) ==
        brzArgMax(h.toBreeze.asInstanceOf[BDV[Double]])) {
        0.0
      }
      else {
        1.0
      }
    }.sum
    sumError / count
  }
}
