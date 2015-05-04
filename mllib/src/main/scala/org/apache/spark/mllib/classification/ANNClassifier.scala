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

import org.apache.spark.mllib.ann.{FeedForwardTrainer, FeedForwardModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.{argmax => Bargmax}

object LabelConverter {

  def apply(labeledPoint: LabeledPoint, labelCount: Int): (Vector, Vector) = {
    val output = Array.fill(labelCount){0.0}
    output(labeledPoint.label.toInt) = 1.0
    (labeledPoint.features, Vectors.dense(output))
  }

  def apply(output: Vector): Double = {
    Bargmax(output.toBreeze.toDenseVector).toDouble
  }
}

class ANNClassifierModel (val annModel: FeedForwardModel)
  extends ClassificationModel with Serializable {
  /**
   * Predict values for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  override def predict(testData: RDD[Vector]): RDD[Double] = testData.map(predict)

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return predicted category from the trained model
   */
  override def predict(testData: Vector): Double = {
    val output = annModel.predict(testData)
    LabelConverter(output)
  }
}

class ANNClassifier (val trainer: FeedForwardTrainer) extends Serializable {

  def train(data: RDD[LabeledPoint]): ANNClassifierModel = {
    // TODO: check that last layer has the needed amount of outputs
    val labeledCount = trainer.outputSize
    val annData = data.map(lp => LabelConverter(lp, labeledCount))
    val model = trainer.train(annData)
    new ANNClassifierModel(model)
  }
}
