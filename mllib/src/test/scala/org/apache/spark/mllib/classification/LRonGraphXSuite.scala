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

import org.apache.spark.mllib.util._
import org.scalatest.{FunSuite, Matchers}

class LRonGraphXSuite extends FunSuite with MLlibTestSparkContext with Matchers {
  test("10M dataSet") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"${sparkHome}/data/mllib/lr_data.10M.txt"
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile)
    LRonGraphX.train(dataSet, 1000, 1e-3)
  }
}
