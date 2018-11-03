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

package org.apache.spark.deploy.k8s.features

import java.util.UUID

import io.fabric8.kubernetes.api.model.HasMetadata

import org.apache.spark.deploy.k8s.{KubernetesConf, KubernetesRoleSpecificConf, SparkPod}

private[spark] class LocalDirsFeatureStep(
  conf: KubernetesConf[_ <: KubernetesRoleSpecificConf],
  defaultLocalDir: String = s"/var/data/spark-${UUID.randomUUID}")
  extends KubernetesFeatureConfigStep {

  val featureStep =
    if (conf.sparkConf.getBoolean("spark.kubernetes.cci.local.dir.evs.enabled", false)) {
      new EvsDirsFeatureStep(conf, defaultLocalDir)
    } else {
      new OriginalLocalDirsFeatureStep(conf, defaultLocalDir)
    }

  def configurePod(pod: SparkPod): SparkPod = featureStep.configurePod(pod)

  def getAdditionalPodSystemProperties(): Map[String, String] =
    featureStep.getAdditionalPodSystemProperties()

  def getAdditionalKubernetesResources(): Seq[HasMetadata] =
    featureStep.getAdditionalKubernetesResources()
}
